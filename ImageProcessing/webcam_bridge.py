#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BRIDGE_PORT = 8090
JPEG_QUALITY = 82
CAPTURE_DELAY_SECONDS = 0.03


class FrameStore:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.running = True

    def set_frame(self, frame: bytes) -> None:
        with self.condition:
            self.frame = frame
            self.condition.notify_all()

    def wait_frame(self, timeout: float = 1.0) -> bytes | None:
        with self.condition:
            if self.frame is None:
                self.condition.wait(timeout)
            return self.frame


def capture_loop(store: FrameStore, camera: str) -> None:
    source: int | str = int(camera) if camera.lstrip("-").isdigit() else camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {camera!r}")

    while store.running:
        ok, frame = cap.read()
        if ok:
            encoded_ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if encoded_ok:
                store.set_frame(encoded.tobytes())
        time.sleep(CAPTURE_DELAY_SECONDS)

    cap.release()


def list_cameras() -> None:
    found = False
    for index in range(11):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found = True
            ok, _ = cap.read()
            status = "ok" if ok else "opens but did not return a frame"
            print(f"{index}: {status}")
        cap.release()
    if not found:
        print("No cameras found.")


def make_handler(store: FrameStore):
    class WebcamHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in {"/", "/video"}:
                self.send_response(404)
                self.end_headers()
                return

            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Use /video for the MJPEG stream.\n")
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            while store.running:
                frame = store.wait_frame()
                if frame is None:
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except OSError:
                    break

        def log_message(self, format: str, *args) -> None:
            return

    return WebcamHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose a local webcam as an MJPEG stream for Docker containers")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or source string")
    parser.add_argument("--list", action="store_true", help="List available camera indexes and exit")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    store = FrameStore()
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(store, args.camera),
        daemon=True,
    )
    capture_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", BRIDGE_PORT), make_handler(store))
    print(f"Webcam stream: http://localhost:{BRIDGE_PORT}/video")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        store.running = False
        server.server_close()


if __name__ == "__main__":
    main()
