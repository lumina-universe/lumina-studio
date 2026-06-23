#!/usr/bin/env python3
"""Lumina 3D AI Server — Flask API + embedded studio UI on port 8765."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

PORT = int(os.environ.get("LUMINA_PORT", "8765"))
HOST = os.environ.get("LUMINA_HOST", "0.0.0.0")
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "lumina-ai-engine",
            "version": "1.0.0",
            "time": datetime.now(timezone.utc).isoformat(),
            "triposr": os.environ.get("LUMINA_TRIPOSR", "stub"),
        }
    )


@app.get("/api/status")
def api_status():
    return jsonify(
        {
            "engine": "TripoSR (stub mode)",
            "jobs": 0,
            "queue": [],
            "outputs": [p.name for p in OUTPUT_DIR.glob("*.json")],
        }
    )


@app.post("/api/reconstruct")
def reconstruct():
    payload = request.get_json(silent=True) or {}
    job_id = str(uuid.uuid4())[:8]
    result = {
        "job_id": job_id,
        "status": "completed",
        "mode": payload.get("mode", "image-to-3d"),
        "message": "Stub reconstruction complete. Mount TripoSR weights for production output.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": 120,
    }
    out = OUTPUT_DIR / f"{job_id}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return jsonify(result), HTTPStatus.CREATED


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


if __name__ == "__main__":
    print(f"Lumina AI Engine listening on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=os.environ.get("LUMINA_DEBUG") == "1")