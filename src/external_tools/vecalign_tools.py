"""Run LASER/Vecalign locally, or explicitly opt into a configured WSL setup."""
from __future__ import annotations

import os
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


DEFAULT_DELETION_PERCENTILE = 0.2


def validate_deletion_percentile(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError('del_percentile_frac must be a finite number between 0 and 1')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('del_percentile_frac must be a finite number between 0 and 1')
    return value


class ExternalToolsRunner:
    def __init__(self, project_dir: Path, *, backend: str | None = None,
                 laser_python: str | None = None, vecalign_python: str | None = None,
                 wsl_project_dir: str | None = None):
        self.project_dir = Path(project_dir).resolve()
        self.backend = backend or os.environ.get('EXTERNAL_TOOLS_BACKEND', 'native')
        if self.backend not in {'native', 'wsl'}:
            raise ValueError('EXTERNAL_TOOLS_BACKEND must be native or wsl')
        self.wsl_project_dir = wsl_project_dir or os.environ.get('WSL_PROJECT_DIR')
        if self.backend == 'wsl' and not self.wsl_project_dir:
            raise ValueError('Set WSL_PROJECT_DIR to the project path inside WSL')
        default_python = 'python3' if self.backend == 'wsl' else sys.executable
        self.laser_python = laser_python or os.environ.get('LASER_PYTHON', default_python)
        self.vecalign_python = vecalign_python or os.environ.get('VECALIGN_PYTHON', default_python)

    def to_wsl_path(self, path: Path) -> str:
        relative = Path(path).resolve().relative_to(self.project_dir)
        if not self.wsl_project_dir:
            raise ValueError('WSL_PROJECT_DIR is not configured')
        return f"{self.wsl_project_dir.rstrip('/')}/{relative.as_posix()}"

    def _path(self, path: Path) -> str:
        path = Path(path)
        if not path.is_absolute():
            path = self.project_dir / path
        return self.to_wsl_path(path) if self.backend == 'wsl' else str(path.resolve())

    def run_command(self, command: list[str], *, stdout=None):
        command = [str(arg) for arg in command]
        if self.backend == 'wsl':
            command = [os.environ.get('WSL_EXECUTABLE', 'wsl.exe'), 'bash', '-lc',
                       f'cd {shlex.quote(self.wsl_project_dir)} && {shlex.join(command)}']
        print('Executing:', shlex.join(command), flush=True)
        return subprocess.run(command, cwd=self.project_dir, check=True, stdout=stdout)

    def _vecalign_entry(self, component: str) -> list[str]:
        # Upstream source checkouts expose scripts; packaged installations may expose modules.
        script = os.environ.get('VECALIGN_SCRIPT' if component == 'vecalign' else 'VECALIGN_OVERLAP_SCRIPT')
        return [self.vecalign_python, script] if script else [self.vecalign_python, '-m', f'vecalign.{component}']

    def generate_overlaps(self, input_path: Path, output_path: Path, num_overlaps: int = 4):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.run_command(self._vecalign_entry('overlap') + [
            '-i', self._path(input_path), '-o', self._path(output_path), '-n', str(num_overlaps)])

    def generate_embeddings(self, input_path: Path, output_path: Path, language: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.run_command([self.laser_python, self._path(self.project_dir / 'src/processing/embedder.py'),
                          self._path(input_path), self._path(output_path), language])

    def align(self, source_path: Path, target_path: Path, source_overlap_path: Path,
              target_overlap_path: Path, source_embeddings_path: Path,
              target_embeddings_path: Path, output_path: Path,
              alignment_max_size: int = 4, one_to_many: bool = False,
              del_percentile_frac: float = DEFAULT_DELETION_PERCENTILE):
        del_percentile_frac = validate_deletion_percentile(del_percentile_frac)
        command = self._vecalign_entry('vecalign') + [
            '-s', self._path(source_path), '-t', self._path(target_path),
            '--src_embed', self._path(source_overlap_path), self._path(source_embeddings_path),
            '--tgt_embed', self._path(target_overlap_path), self._path(target_embeddings_path),
            '-a', str(alignment_max_size), '--del_percentile_frac', str(del_percentile_frac)]
        if one_to_many:
            command.append('--one_to_many')
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=output_path.parent, delete=False) as output:
                temporary = Path(output.name)
                self.run_command(command, stdout=output)
            os.replace(temporary, output_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f'Alignment saved: {output_path}', flush=True)
