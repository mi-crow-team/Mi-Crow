import importlib


def test_import_amber_succeeds_and_has_docstring():
    mod = importlib.import_module("amber")
    # Basic sanity checks to ensure the package can be imported and has a docstring
    assert mod.__name__ == "amber"
    assert isinstance(mod.__doc__, str)
    assert "Amber" in (mod.__doc__ or "")


def test_package_name_and_ping():
    amber = importlib.import_module("amber")
    assert getattr(amber, "PACKAGE_NAME", None) == "amber"
    assert amber.ping() == "pong"
