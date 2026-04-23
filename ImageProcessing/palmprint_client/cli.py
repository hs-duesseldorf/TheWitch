from __future__ import annotations

from contextlib import suppress

from .config import parse_args, prepare_runtime_environment
from .utils import get_access_urls, set_deterministic


def main() -> None:
    config = parse_args()
    prepare_runtime_environment()

    import uvicorn

    from .debug_server import create_debug_app
    from .runtime import HeadlessPalmClient
    from .transport import LatestMessageBus, MessageQueue, PipelineWebSocketClient

    set_deterministic(config.seed)

    event_bus = LatestMessageBus()
    feature_queue = MessageQueue(maxlen=256)
    pipeline_client = PipelineWebSocketClient(config.transport.pipeline_ws_url, feature_queue)

    runtime = HeadlessPalmClient(
        config.runtime,
        event_bus=event_bus,
        feature_queue=feature_queue,
        pipeline_status_provider=pipeline_client.snapshot,
    )
    try:
        runtime.start()
        pipeline_client.start()

        app = create_debug_app(event_bus=event_bus)
        access_urls = get_access_urls(config.server.host, config.server.port)

        print("Palmprint FastAPI debug server is serving on:")
        for url in access_urls:
            print(f"  {url}")
        print(f"Upstream AI websocket target: {config.transport.pipeline_ws_url}")
        print("If the upstream server is offline, capture and the local debug UI continue running.")

        server = uvicorn.Server(
            uvicorn.Config(
                app=app,
                host=config.server.host,
                port=config.server.port,
                log_level="info",
            )
        )
        with suppress(KeyboardInterrupt):
            server.run()
    finally:
        pipeline_client.stop()
        runtime.stop()
