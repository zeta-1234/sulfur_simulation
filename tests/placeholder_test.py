from __future__ import annotations


def test_import() -> None:
    """Test that the phonon_simulation package can be imported."""
    try:
        import phonon_simulation  # noqa: PLC0415
    except ImportError:
        phonon_simulation = None

    assert phonon_simulation is not None, "phonon_simulation module should not be None"
