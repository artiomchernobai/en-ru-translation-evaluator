import shlex
import subprocess
from pathlib import Path


class ExternalToolsRunner:

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

        self.wsl_project_dir = "/mnt/d/piton"

        self.laser_python = (
            "/home/artiom/tools/laser-env/bin/python"
        )

        self.vecalign_python = (
            "/home/artiom/tools/vecalign-env/bin/python"
        )

    def to_wsl_path(self, path: Path) -> str:
        path = Path(path).resolve()
        relative_path = path.relative_to(self.project_dir)

        return f"{self.wsl_project_dir}/{relative_path.as_posix()}"

    def run_command(self, command: list[str]):
        full_command = [
            "wsl",
            "bash",
            "-lc",
            f"cd {self.wsl_project_dir} && "
            + " ".join(command)
        ]

        print("Executing:", " ".join(full_command))

        subprocess.run(
            full_command,
            cwd=self.project_dir,
            check=True,
        )

    def generate_overlaps(
            self,
            input_path: Path,
            output_path: Path,
            num_overlaps: int = 4,
    ):
        self.run_command([
            self.vecalign_python,
            "-m",
            "vecalign.overlap",
            "-i",
            self.to_wsl_path(input_path),
            "-o",
            self.to_wsl_path(output_path),
            "-n",
            str(num_overlaps),
        ])

    def generate_embeddings(
        self,
        input_path: Path,
        output_path: Path,
        language: str,
    ):
        self.run_command([
            str(self.laser_python),
            "src/processing/embedder.py",
            str(self.to_wsl_path(input_path)),
            str(self.to_wsl_path(output_path)),
            language,
        ])

    def align(
            self,
            source_path: Path,
            target_path: Path,
            source_overlap_path: Path,
            target_overlap_path: Path,
            source_embeddings_path: Path,
            target_embeddings_path: Path,
            output_path: Path,
            alignment_max_size: int = 4,
            one_to_many: bool = False
    ):
        command = [
            self.vecalign_python,
            "-m", "vecalign.vecalign",

            "-s", self.to_wsl_path(source_path),
            "-t", self.to_wsl_path(target_path),

            "--src_embed",
            self.to_wsl_path(source_overlap_path),
            self.to_wsl_path(source_embeddings_path),

            "--tgt_embed",
            self.to_wsl_path(target_overlap_path),
            self.to_wsl_path(target_embeddings_path),

            "-a", str(alignment_max_size),
        ]

        if one_to_many:
            command.append("--one_to_many")

        linux_command = " ".join(
            shlex.quote(str(arg)) for arg in command
        )

        full_command = [
            "wsl.exe",
            "bash",
            "-lc",
            f"cd {shlex.quote(self.wsl_project_dir)} && {linux_command}"
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as output_file:
            subprocess.run(
                full_command,
                stdout=output_file,
                check=True,
                text=True
            )

        print(f"Alignment saved: {output_path}")