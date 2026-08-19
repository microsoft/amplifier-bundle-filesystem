"""Tests for the V4A diff helper.

Ported from the OpenAI Agents SDK (Apache-2.0).
Source: https://github.com/openai/openai-agents-python/blob/main/tests/test_apply_diff.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amplifier_module_tool_apply_patch.apply_diff import (
    apply_diff,
    read_source,
    write_source,
)

_BOM = b"\xef\xbb\xbf"


def test_apply_diff_with_floating_hunk_adds_lines() -> None:
    diff = "\n".join(["@@", "+hello", "+world"])  # no trailing newline
    assert apply_diff("", diff) == "hello\nworld\n"


def test_apply_diff_with_empty_input_and_crlf_diff_preserves_crlf() -> None:
    diff = "\r\n".join(["@@", "+hello", "+world"])
    assert apply_diff("", diff) == "hello\r\nworld\r\n"


def test_apply_diff_create_mode_requires_plus_prefix() -> None:
    diff = "plain line"
    with pytest.raises(ValueError):
        apply_diff("", diff, mode="create")


def test_apply_diff_create_mode_error_message_describes_missing_prefix() -> None:
    """Error message should say 'missing +' prefix', not 'Invalid Add File Line'."""
    with pytest.raises(ValueError, match=r"missing '\+' prefix"):
        apply_diff("", "plain line without prefix", mode="create")


def test_apply_diff_create_mode_perserves_trailing_newline() -> None:
    diff = "\n".join(["+hello", "+world", "+"])
    assert apply_diff("", diff, mode="create") == "hello\nworld\n"


def test_apply_diff_applies_contextual_replacement() -> None:
    input_text = "line1\nline2\nline3\n"
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff(input_text, diff) == "line1\nupdated\nline3\n"


def test_apply_diff_raises_on_context_mismatch() -> None:
    input_text = "one\ntwo\n"
    diff = "\n".join(["@@ -1,2 +1,2 @@", " x", "-two", "+2"])
    with pytest.raises(ValueError):
        apply_diff(input_text, diff)


def test_apply_diff_with_crlf_input_and_lf_diff_preserves_crlf() -> None:
    input_text = "line1\r\nline2\r\nline3\r\n"
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff(input_text, diff) == "line1\r\nupdated\r\nline3\r\n"


def test_apply_diff_with_lf_input_and_crlf_diff_preserves_lf() -> None:
    input_text = "line1\nline2\nline3\n"
    diff = "\r\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff(input_text, diff) == "line1\nupdated\nline3\n"


def test_apply_diff_with_crlf_input_and_crlf_diff_preserves_crlf() -> None:
    input_text = "line1\r\nline2\r\nline3\r\n"
    diff = "\r\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff(input_text, diff) == "line1\r\nupdated\r\nline3\r\n"


def test_apply_diff_create_mode_preserves_crlf_newlines() -> None:
    diff = "\r\n".join(["+hello", "+world", "+"])
    assert apply_diff("", diff, mode="create") == "hello\r\nworld\r\n"


def test_apply_diff_with_cr_only_input_applies_cleanly() -> None:
    """Classic-Mac (CR-only) files must still patch.

    read_source decodes raw bytes with no universal-newline translation, so a
    lone \\r now reaches the parser verbatim. Without folding it to \\n the whole
    file collapses into a single line and context matching fails outright.
    """
    input_text = "line1\rline2\rline3\r"
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])
    assert apply_diff(input_text, diff) == "line1\nupdated\nline3\n"


# --- read_source / write_source byte fidelity -------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"line1\nline2\n", id="lf"),
        pytest.param(b"line1\r\nline2\r\n", id="crlf"),
        pytest.param(b"line1\rline2\r", id="cr"),
        pytest.param(b"line1\nline2\r\nline3\n", id="mixed"),
        pytest.param(_BOM + b"line1\nline2\n", id="bom-lf"),
        pytest.param(_BOM + b"line1\r\nline2\r\n", id="bom-crlf"),
        pytest.param(b"no trailing newline", id="no-trailing-newline"),
        pytest.param(b"", id="empty"),
    ],
)
def test_read_write_source_round_trip_is_byte_exact(tmp_path: Path, raw: bytes) -> None:
    """read_source -> write_source must be a byte-for-byte identity.

    This is the contract the whole fix rests on: no newline translation in
    either direction, and a BOM restored only when one was there to begin with.
    """
    path = tmp_path / "sample.txt"
    path.write_bytes(raw)

    text, had_bom = read_source(path)
    write_source(path, text, had_bom)

    assert path.read_bytes() == raw


def test_read_source_strips_bom_from_returned_text(tmp_path: Path) -> None:
    """The BOM must not survive into the text, or it glues U+FEFF onto line 1."""
    path = tmp_path / "bom.txt"
    path.write_bytes(_BOM + b"line1\nline2\n")

    text, had_bom = read_source(path)

    assert had_bom is True
    assert text == "line1\nline2\n"


def test_read_source_reports_no_bom_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"
    path.write_bytes(b"line1\nline2\n")

    text, had_bom = read_source(path)

    assert had_bom is False
    assert text == "line1\nline2\n"


def test_write_source_does_not_invent_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "plain.txt"

    write_source(path, "line1\nline2\n")

    assert path.read_bytes() == b"line1\nline2\n"


def test_write_source_preserves_crlf_without_translation(tmp_path: Path) -> None:
    """Text-mode writes translate \\n -> \\r\\n on Windows; write_bytes must not."""
    path = tmp_path / "crlf.txt"

    write_source(path, "line1\r\nline2\r\n")

    assert path.read_bytes() == b"line1\r\nline2\r\n"


# --- end-to-end: read -> patch -> write preserves the file's own style ------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(b"line1\nline2\nline3\n", b"line1\nupdated\nline3\n", id="lf"),
        pytest.param(
            b"line1\r\nline2\r\nline3\r\n",
            b"line1\r\nupdated\r\nline3\r\n",
            id="crlf",
        ),
        pytest.param(
            _BOM + b"line1\nline2\nline3\n",
            _BOM + b"line1\nupdated\nline3\n",
            id="bom-lf",
        ),
        pytest.param(
            _BOM + b"line1\r\nline2\r\nline3\r\n",
            _BOM + b"line1\r\nupdated\r\nline3\r\n",
            id="bom-crlf",
        ),
        pytest.param(b"line1\rline2\rline3\r", b"line1\nupdated\nline3\n", id="cr"),
    ],
)
def test_patch_round_trip_preserves_newline_style_and_bom(
    tmp_path: Path, raw: bytes, expected: bytes
) -> None:
    """A one-line patch must not rewrite the rest of the file's line endings.

    The reported bug: on Windows a CRLF file round-tripped through read_text /
    write_text lost its style and came back rewritten end to end. Asserting on
    bytes makes that regression fail here too, not only on Windows.
    """
    path = tmp_path / "sample.txt"
    path.write_bytes(raw)
    diff = "\n".join(["@@ line1", "-line2", "+updated", " line3"])

    text, had_bom = read_source(path)
    write_source(path, apply_diff(text, diff), had_bom)

    assert path.read_bytes() == expected
