from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_intelligence.api.app import create_app
from query_intelligence.chatbot import apply_live_data_env, load_chatbot_config


def open_browser_when_ready(url: str, health_url: str, *, timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    open_browser(url)
                    return
        except Exception:
            time.sleep(1)
    print(f"Server readiness check timed out. Trying to open browser anyway: {url}", flush=True)
    open_browser(url)


def open_browser(url: str) -> None:
    print(f"Opening browser: {url}", flush=True)
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)  # noqa: S606
        else:
            webbrowser.open(url)
    except Exception as exc:  # noqa: BLE001
        print(f"Automatic browser launch failed: {exc}", flush=True)
        print(f"Please open this address manually: {url}", flush=True)
        webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local Financial Chatbot web app.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "app_config.json")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser.")
    parser.add_argument("--host", help="Override configured host.")
    parser.add_argument("--port", type=int, help="Override configured port.")
    args = parser.parse_args()

    launch_started_at = time.perf_counter()
    print("[startup] Loading configuration...", flush=True)
    config = load_chatbot_config(args.config)
    if args.host:
        config["server"]["host"] = args.host
    if args.port:
        config["server"]["port"] = args.port
    apply_live_data_env(config)
    print("[startup] Configuration loaded. Live data defaults are enabled.", flush=True)

    host = str(config["server"]["host"])
    port = int(config["server"]["port"])
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}/"
    health_url = f"http://{browser_host}:{port}/health"

    print(f"Financial Chatbot is starting at {url}", flush=True)
    print("Keep this command window open while using the chatbot.", flush=True)
    print(
        "[startup] Initializing Query Intelligence pipelines and runtime data. "
        "The first launch can take a while; the browser opens after the health check passes.",
        flush=True,
    )
    app_started_at = time.perf_counter()
    app = create_app(app_config=config)
    print(f"[startup] Backend initialized in {time.perf_counter() - app_started_at:.1f}s.", flush=True)

    if not args.no_browser:
        print(f"[startup] Browser will open automatically when the server is ready: {url}", flush=True)
        threading.Thread(
            target=open_browser_when_ready,
            args=(url, health_url),
            daemon=True,
        ).start()
    else:
        print(f"[startup] Browser auto-open disabled. Visit after startup: {url}", flush=True)

    print(f"[startup] Starting web server on {url}...", flush=True)
    print(f"If the browser does not open automatically after startup completes, visit: {url}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
    print(f"[startup] Server stopped after {time.perf_counter() - launch_started_at:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
