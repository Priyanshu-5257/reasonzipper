from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_filler_thinking.py"
    spec = importlib.util.spec_from_file_location("make_filler_thinking", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_filler_completion_shape():
    mod = _load()
    text = mod.build_filler_completion(4, ".", 2)
    assert text.startswith("<think>\n")
    assert text.endswith("ERROR_LINE=2")
    assert "\n</think>\n" in text
    assert text.count(".") >= 4
