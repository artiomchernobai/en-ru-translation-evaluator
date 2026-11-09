"""Subprocess jobs and incremental log files, shared by GUI and worker."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .config import DOCUMENTS, STAGES, folder
from src.external_tools.vecalign_tools import DEFAULT_DELETION_PERCENTILE, validate_deletion_percentile


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def job_paths(root: Path, job_id: str) -> tuple[Path, Path, Path]:
    directory = folder(root, "jobs")
    return (directory / f"{job_id}.request.json",
            directory / f"{job_id}.status.json",
            directory / f"{job_id}.log")


def start_job(root: Path, stage: str, document_names: list[str], max_size: int,
              del_percentile_frac: float = DEFAULT_DELETION_PERCENTILE) -> dict:
    """Start one isolated worker. The UI caller enforces one live job per session."""
    if stage not in (*STAGES, "all"):
        raise ValueError(f"Unknown stage: {stage}")
    if not document_names or any(doc not in DOCUMENTS for doc in document_names):
        raise ValueError("Выберите существующие документы")
    if max_size not in (2, 3, 4):
        raise ValueError("max_size должен быть 2, 3 или 4")

    del_percentile_frac = validate_deletion_percentile(del_percentile_frac)
    job_id = uuid.uuid4().hex[:12]
    request_path, status_path, log_path = job_paths(root, job_id)
    write_json(request_path, {
        "id": job_id,
        "stage": stage,
        "documents": list(dict.fromkeys(document_names)),
        "max_size": max_size,
        "del_percentile_frac": del_percentile_frac,
        "project_root": str(root.resolve()),
    })
    write_json(status_path, {"state": "queued", "stage": stage, "started": time.time()})
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.ui.worker", "--job-file", str(request_path)],
            cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    return {"id": job_id, "process": process, "stage": stage}


def read_status(root: Path, job_id: str) -> dict:
    _, path, _ = job_paths(root, job_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"state": "unknown"}


def tail_log(root: Path, job_id: str, max_bytes: int = 40_000) -> str:
    _, _, path = job_paths(root, job_id)
    if not path.exists():
        return "Пока нет сообщений."
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(stream.tell() - max_bytes, 0))
        return stream.read().decode("utf-8", errors="replace")


def cancel_job(root: Path, job: dict) -> None:
    process = job.get("process")
    if process and process.poll() is None:
        process.terminate()
        _, status_path, _ = job_paths(root, job["id"])
        write_json(status_path, {"state": "cancelled", "stage": job["stage"], "ended": time.time()})
