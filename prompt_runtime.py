"""Hot-reloadable initial_prompt for a pinned Wyoming Faster Whisper server."""
import asyncio
import hmac
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_PROMPT_LENGTH = 16000
MAX_BODY_BYTES = 100000


class PromptStore:
    def __init__(self, path, default=""):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._prompt = default
        if self.path.exists():
            self._prompt = self.validate(json.loads(self.path.read_text(encoding="utf-8")))

    @staticmethod
    def validate(payload):
        if isinstance(payload, dict) and isinstance(payload.get("initial_prompt"), list):
            phrases = payload["initial_prompt"]
            if not all(isinstance(phrase, str) for phrase in phrases):
                raise ValueError("initial_prompt must contain strings")
            payload = {"initial_prompt": ";".join(phrases)}
        if not isinstance(payload, dict) or not isinstance(payload.get("initial_prompt"), str):
            raise ValueError("initial_prompt must be a string")
        value = payload["initial_prompt"]
        if len(value) > MAX_PROMPT_LENGTH:
            raise ValueError("initial_prompt is too long")
        return value

    def get(self):
        with self._lock:
            return self._prompt

    def set(self, prompt):
        self.validate({"initial_prompt": prompt})
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".initial-prompt-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump({"initial_prompt": prompt}, stream, ensure_ascii=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, self.path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            self._prompt = prompt


def install_live_prompt(loader_class, store):
    """Keep upstream transcription code; only its prompt property becomes dynamic."""
    loader_class.initial_prompt = property(lambda self: store.get(), lambda self, value: None)


def make_server(address, token, store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            # Do not put prompts, tokens or request paths in access logs.
            pass

        def respond(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()):
                self.respond(401, {"error": "unauthorized"})
                return False
            if self.path != "/initial-prompt":
                self.respond(404, {"error": "not_found"})
                return False
            return True

        def do_GET(self):
            if self.authorized():
                self.respond(200, {"initial_prompt": store.get()})

        def do_PUT(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    self.respond(413, {"error": "invalid_body_size"})
                    return
                self.connection.settimeout(10)
                payload = json.loads(self.rfile.read(length))
                prompt = store.validate(payload)
                store.set(prompt)
            except (ValueError, UnicodeError):
                self.respond(400, {"error": "invalid_prompt"})
                return
            except OSError:
                self.respond(500, {"error": "storage_failed"})
                return
            result = payload["initial_prompt"] if isinstance(payload["initial_prompt"], list) else prompt
            self.respond(200, {"initial_prompt": result, "applied": True})

    return ThreadingHTTPServer(address, Handler)


def main():
    from wyoming_faster_whisper.models import ModelLoader
    from wyoming_faster_whisper.__main__ import main as whisper_main

    default = Path(__file__).with_name("default-initial-prompt.txt").read_text(encoding="utf-8").strip()
    store = PromptStore(os.getenv("WHISPER_PROMPT_FILE", "/data/initial-prompt.json"), default)
    install_live_prompt(ModelLoader, store)
    token = os.getenv("WHISPER_PROMPT_TOKEN", "")
    server = make_server((os.getenv("WHISPER_PROMPT_HOST", "0.0.0.0"), int(os.getenv("WHISPER_PROMPT_PORT", "10301"))), token, store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        asyncio.run(whisper_main())
    finally:
        if server:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
