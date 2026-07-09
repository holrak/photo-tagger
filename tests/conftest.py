"""Shared pytest fixtures for the photo-tagger test suite."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic_ai import BinaryContent


if TYPE_CHECKING:
    from pydantic_ai import Agent


@pytest.fixture(autouse=True)
def _isolate_telemetry_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Keep telemetry hermetic: redirect its state dir to tmp and clear the opt-out env vars.

    ``main.tag`` and the GUI consult telemetry on every run, so without this the suite would read
    and write the developer's real ``~/.local/state/photo-tagger`` and be swayed by a stray
    ``DO_NOT_TRACK`` in their shell. No network send happens here: the beacon fires from
    ``run_batch``'s completion callback, which the CLI tests mock out.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "pt-state"))
    monkeypatch.delenv("PHOTO_TAGGER_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)


@pytest.fixture
def stub_jpeg_bytes() -> BinaryContent:
    """Return a tiny placeholder JPEG used to short-circuit image preparation in unit tests."""
    return BinaryContent(data=b"\xff\xd8stub", media_type="image/jpeg")


@pytest.fixture
def fake_agent() -> Agent:
    """Return a typed-but-bogus Agent placeholder for tests that mock every IO collaborator."""
    return cast("Agent", object())
