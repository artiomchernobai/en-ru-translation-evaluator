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

from src.processing.pipeline.config import DOCUMENTS, STAGES, TARGETS, folder
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


def start_job(root: Path, stage: str, document_names: list[str], max_size: int | None = None,
              del_percentile_frac: float | None = None, selection: dict | None = None) -> dict:
    """Start one isolated worker. The UI caller enforces one live job per session."""
    from src.processing.pipeline.settings import read_settings
    settings = read_settings(root)
    max_size = settings['vecalign']['max_size'] if max_size is None else max_size
    del_percentile_frac = settings['vecalign']['deletion_percentile'] if del_percentile_frac is None else del_percentile_frac
    if selection is not None:
        if not isinstance(selection,dict):raise ValueError('Invalid candidate selection')
        if stage!='mqm' or not {'group_index','target'}<=set(selection) or set(selection)-{'group_index','target','strategy'} or type(selection['group_index']) is not int or selection['group_index']<0 or selection['target'] not in TARGETS:
            raise ValueError('Invalid candidate selection')
        if selection.get('strategy','fixed') not in ('fixed','adaptive'):raise ValueError('Invalid candidate strategy')
    if stage not in (*STAGES, "all"):
        raise ValueError(f"Unknown stage: {stage}")
    if not document_names or any(doc not in DOCUMENTS for doc in document_names):
        raise ValueError("Выберите существующие документы")
    if type(max_size) is not int or not 2 <= max_size <= 8:
        raise ValueError("max_size должен быть от 2 до 8")

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
        "selection": selection,
    })
    write_json(status_path, {"state": "queued", "stage": stage, "started": time.time()})
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
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
