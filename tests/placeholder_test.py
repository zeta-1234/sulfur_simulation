from __future__ import annotations


def test_import() -> None:
    try:
        import sulfur_simulation  # noqa: PLC0415
    except ImportError:
        sulfur_simulation = None

    assert sulfur_simulation is not None, "sulfur_simulation module should not be None"
