from __future__ import annotations

import csv

from pathlib import Path

from pzid.optimize import LMOptions, LMResult, solve_underdetermined_lm

def write_history(path: Path, result: LMResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "iteration",
                "cost",
                "residual_norm",
                "step_norm",
                "damping",
                "rejected_attempts",
            ),
        )
        writer.writeheader()
        for iteration in result.history:
            writer.writerow(iteration.__dict__)
