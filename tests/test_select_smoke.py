"""Import / preassigned-path smoke tests (no GPU)."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_select_script_imports_build_think_completion():
    from reasonzipper.existing_vocab import (
        assert_original_vocabulary,
        build_candidate_token_ids,
        build_think_completion_from_ids,
    )

    assert callable(build_think_completion_from_ids)
    assert callable(build_candidate_token_ids)
    assert callable(assert_original_vocabulary)


def test_select_script_parses_and_has_no_bare_result_after_preassigned():
    """Guard: preassigned path must not read unbound ``result``."""
    src = (ROOT / "scripts" / "select_compressed_codes.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Ensure the symbol is importable from the module text
    assert "build_think_completion_from_ids" in src
    assert "score_prefix_answer_nll" in src
    # The rewritten path uses ``selected`` / ``result_meta``, not a shared ``result``
    # after both branches.
    assert "result_meta" in src
    # Fail if we reintroduce the old bug pattern outside the select branch.
    # Allow ``selected[...]`` but disallow standalone ``result[`` / ``result.get``.
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "result":
                raise AssertionError(
                    "select_compressed_codes.py still indexes bare name 'result'"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "result"
                and node.func.attr == "get"
            ):
                raise AssertionError(
                    "select_compressed_codes.py still calls result.get(...)"
                )


def test_eval_model_does_not_unconditionally_resize():
    src = (ROOT / "scripts" / "eval_model.py").read_text(encoding="utf-8")
    assert "refuse_embedding_resize" in src
    assert "keeping padded embeddings" in src
    # Must not contain the old unconditional shrink pattern alone
    assert "if len(tokenizer) != model.get_input_embeddings()" not in src
