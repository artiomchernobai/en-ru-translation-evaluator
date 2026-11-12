"""Isolated task entrypoint: python -u -m src.ui.worker --job-file ..."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from src.processing.pipeline.config import folder
from src.external_tools.vecalign_tools import DEFAULT_DELETION_PERCENTILE
from .jobs import write_json
from src.processing.pipeline.runner import run_stage


def run_job(request_file: Path) -> int:
    request = json.loads(request_file.read_text(encoding="utf-8"))
    root = Path(request["project_root"]).resolve()
    # The file must belong to the project itself, not an arbitrary path.
    if request_file.resolve().parent != folder(root, "jobs").resolve():
        raise ValueError("Job request outside project data/ui_jobs")
    job_id = request["id"]
    status_path = folder(root, "jobs") / f"{job_id}.status.json"
    status = {"state": "running", "stage": request["stage"], "started": time.time()}
    write_json(status_path, status)
    print(f"Задача {job_id} запущена. Проект: {root}", flush=True)
    try:
        run_stage(root, request["stage"], request["documents"], int(request["max_size"]),
                  del_percentile_frac=request.get("del_percentile_frac", DEFAULT_DELETION_PERCENTILE))
    except Exception:
        print("\nОШИБКА:", flush=True)
        traceback.print_exc()
        write_json(status_path, {**status, "state": "error", "ended": time.time()})
        return 1
    write_json(status_path, {**status, "state": "done", "ended": time.time()})
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_job(args.job_file))


if __name__ == "__main__":
    main()
