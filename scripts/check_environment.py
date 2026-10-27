#!/usr/bin/env python3
"""Record runtime environment for reproducibility."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _pkg_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def _git_sha(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=root / "artifacts/reports/environment.json")
    args = p.parse_args()

    info: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(root),
        "packages": {
            name: _pkg_version(name)
            for name in [
                "torch",
                "transformers",
                "datasets",
                "peft",
                "trl",
                "accelerate",
                "bitsandbytes",
                "yaml",
                "numpy",
            ]
        },
    }

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["gpus"] = []
        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
    except Exception as e:
        info["torch_error"] = str(e)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
