# app/main.py
import os, sys, asyncio
from typing import Any, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from lmnr import Laminar, Instruments
# ----- Windows event loop fix (prevents weird crashes with browsers) -----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass

# ----- Load env / keys -----
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LMNR_PROJECT_API_KEY = os.getenv("LMNR_PROJECT_API_KEY")
Laminar.initialize(project_api_key=LMNR_PROJECT_API_KEY, disabled_instruments={Instruments.BROWSER_USE})

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment. Create a .env with GOOGLE_API_KEY=")

MODEL_NAME = os.getenv("MODEL", "gemini-2.5-flash")

# ----- Agent deps (support both new/old browser_use) -----
try:
    # newer API
    from browser_use import Agent, Browser, ChatGoogle
except ImportError:
    # older API name
    from browser_use import Agent, ChatGoogleGenerativeAI as ChatGoogle, Browser  # type: ignore

app = FastAPI(title="Researcher Agent using Browser Use")

# ----------------------------------------------------
# CORS
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Models
# ----------------------------------------------------
class RunRequest(BaseModel):
    topic: str
    max_results: int = 3      # kept for task templating (ignored in response)
    max_steps: int = 30
    headless: bool = True

class RunResponse(BaseModel):
    ok: bool
    message: str
    data: Dict[str, Any]   # will only contain {"final_result": "..."}

@app.get("/health")
async def health():
    return {"ok": True}

# ----- helpers -----
def _read_task_template(topic: str, max_results: int) -> str:
    """
    Reads taskgpt.txt from the same folder as this file OR ./taskgpt.txt (project root fallback).
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "taskgpt.txt",
        Path.cwd() / "taskgpt.txt",
        Path.cwd() / "app" / "taskgpt.txt",
    ]
    for p in candidates:
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            return txt.replace("{topic}", topic).replace("{max_results}", str(max_results))
    raise FileNotFoundError(
        "Could not find taskgpt.txt. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )

def _extract_final_text(result: Any) -> str:
    """
    Tries to pull a human-readable final result from various browser_use result shapes.
    """
    # attribute callables or values
    keys = ("final_result", "text", "content", "output", "result", "extracted_content")
    for k in keys:
        v = getattr(result, k, None)
        if v is None:
            continue
        try:
            v = v() if callable(v) else v
        except Exception:
            pass
        if v is None:
            continue
        # stringify dict/list nicely
        if isinstance(v, (dict, list)):
            import json
            return json.dumps(v, ensure_ascii=False, indent=2)
        return str(v)

    # fallback
    return str(result)

async def run_agent(topic: str, max_results: int, max_steps: int, headless: bool) -> Dict[str, Any]:
    # Create LLM and Browser
    llm = ChatGoogle(model=MODEL_NAME)

    try:
        browser = Browser(
            window_size={"width": 1200, "height": 800},
            headless=headless,
            keep_alive=False,
            # These kwargs are ignored by older versions, harmless if unsupported:
            #browser_type=os.getenv("BROWSER_USE_PLAYWRIGHT_BROWSER", "chromium"),
            use_cloud = True
            api_key=os.getenv("BROWSER_USE_API_KEY", None)
        )
    except Exception as e:
        # Very common: Playwright not installed or Chromium missing
        raise RuntimeError(
            "Failed to start Browser. If this is a fresh env, run:\n"
            "  pip install playwright\n"
            "  python -m playwright install\n"
            "If on Linux, you may need:  python -m playwright install --with-deps\n"
            f"Original error: {e}"
        )

    # Load your task template (robust pathing)
    task = _read_task_template(topic, max_results)
    agent = Agent(task=task, llm=llm, browser=browser)

    try:
        # Some versions support max_steps; if not, this will be ignored gracefully
        result = await agent.run(max_steps=max_steps)
        final_text = _extract_final_text(result)
        return {"final_result": final_text}
    finally:
        # Close browser safely across versions
        for name in ("close", "shutdown", "quit"):
            fn = getattr(browser, name, None)
            if fn:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        await fn()
                    else:
                        fn()
                except Exception:
                    pass

@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest):
    try:
        data = await run_agent(
            request.topic, request.max_results, request.max_steps, request.headless
        )
        return RunResponse(ok=True, message="done", data=data)
    except Exception as e:
        # Surface the root cause to the client to speed up debugging
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")
