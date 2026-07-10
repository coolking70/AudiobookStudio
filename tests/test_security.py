from pathlib import Path

import pytest

from security import SecurityError, resolve_within, validate_remote_url


def test_resolve_within_rejects_parent_escape(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()
    assert resolve_within("clips/a.wav", root) == (root / "clips/a.wav").resolve()
    with pytest.raises(SecurityError):
        resolve_within("../secret.txt", root)


def test_remote_url_blocks_unsafe_schemes_and_metadata():
    with pytest.raises(SecurityError):
        validate_remote_url("file:///etc/passwd")
    with pytest.raises(SecurityError):
        validate_remote_url("http://169.254.169.254/latest/meta-data")
    assert validate_remote_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
