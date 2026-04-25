#!/usr/bin/env python3

import argparse
from contextlib import suppress

import uvicorn

from palmprint_client.config import prepare_runtime_environment
from palmprint_client.debug_server import create_debug_app
from palmprint_client.utils import get_access_urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Palmprint debug websocket server")
    parser.add_argument("--port", type=int, default=8001, help="FastAPI bind port (default: 8001)")
    args = parser.parse_args()

    if args.port <= 0:
        parser.error("--port must be > 0")

    prepare_runtime_environment()
    app = create_debug_app()

    # Print access URLs
    loopback = f"http://127.0.0.1:{args.port}"
    urls = get_access_urls("0.0.0.0", args.port)
    access_urls = [f"http://localhost:{args.port}" if url == loopback else url for url in urls]

    print("Palmprint debug UI:")
    for url in access_urls:
        print(f"  {url}")

    ws_url = access_urls[0].replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    print(f"Jetson runtime websocket URL: {ws_url}/ws/palmprint")

    # Start server
    server = uvicorn.Server(
        uvicorn.Config(app=app, host="0.0.0.0", port=args.port, log_level="warning")
    )
    with suppress(KeyboardInterrupt):
        server.run()


if __name__ == "__main__":
    main()
