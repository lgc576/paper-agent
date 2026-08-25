from __future__ import annotations

import argparse
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingService:
    def __init__(self, model_path: Path, batch_size: int) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path
        self.batch_size = batch_size
        self.model = SentenceTransformer(str(model_path), device="cpu")

    def embed(self, texts: list[str], dimensions: int | None) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        results: list[list[float]] = []
        for vector in vectors:
            values = [float(value) for value in vector.tolist()]
            if dimensions is not None:
                if dimensions <= 0:
                    raise ValueError("dimensions must be positive")
                if dimensions > len(values):
                    raise ValueError(f"requested dimensions {dimensions} exceeds native dimension {len(values)}")
                values = _normalize(values[:dimensions])
            results.append(values)
        return results


class EmbeddingHandler(BaseHTTPRequestHandler):
    service: EmbeddingService

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "service": "local-embedding",
                    "model": MODEL_ID,
                    "model_path": str(self.service.model_path),
                },
            )
            return
        if self.path.rstrip("/") == "/v1/models":
            _json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        _json_response(self, 404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/embeddings":
            _json_response(self, 404, {"error": {"message": "not found"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8") if body else "{}")
            raw_input = payload.get("input", "")
            dimensions = payload.get("dimensions")
            if isinstance(raw_input, str):
                texts = [raw_input]
            elif isinstance(raw_input, list):
                texts = [str(item) for item in raw_input]
            else:
                raise ValueError("input must be a string or list of strings")
            vectors = self.service.embed(texts, dimensions)
            _json_response(
                self,
                200,
                {
                    "object": "list",
                    "model": payload.get("model") or MODEL_ID,
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": vector,
                        }
                        for index, vector in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 400, {"error": {"message": str(exc), "type": "invalid_request_error"}})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible embedding server")
    parser.add_argument("--host", default=os.getenv("EMBEDDING_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("EMBEDDING_PORT", "8001")))
    parser.add_argument(
        "--model-path",
        default=os.getenv("EMBEDDING_MODEL_PATH", str(_project_root() / "models" / "Qwen" / "Qwen3-Embedding-0.6B")),
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "4")))
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise SystemExit(f"Embedding model path does not exist: {model_path}")

    EmbeddingHandler.service = EmbeddingService(model_path=model_path, batch_size=args.batch_size)
    server = ThreadingHTTPServer((args.host, args.port), EmbeddingHandler)
    print(f"Local embedding server running at http://{args.host}:{args.port}/v1")
    print(f"Model: {MODEL_ID}")
    server.serve_forever()


if __name__ == "__main__":
    main()
