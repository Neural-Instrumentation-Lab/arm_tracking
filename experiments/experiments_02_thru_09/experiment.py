from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Experiment:
    name: str                # short, unique, CLI-friendly id (e.g. "baseline_v1")
    description: str         # one-line human explanation, shown in --list
    trajectory: Path         # the 1 trajectory file
    actualArm: Path              # the 2 arm files
    illusoryArm: Path
    results: Path # where .pkl of results is stored
    nDof : int # how many degrees of freedom arm has

    def validate(self) -> list[str]:
        """Return a list of problems (missing files etc). Empty list = OK."""
        problems = []
        for label, path in [
            ("trajectory", self.trajectory),
            ("actualArm", self.actualArm),
            ("illusoryArm", self.illusoryArm),
        ]:
            if not path.exists():
                problems.append(f"{label} file not found: {path}")
        return problems
