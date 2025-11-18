
def pytest_addoption(parser):
    parser.addoption(
        "--unit",
        action="store_true",
        default=False,
        help="Run only unit tests (from tests/unit/)",
    )
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run only end-to-end tests (from tests/e2e/)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--unit"):
        selected = []
        deselected = []
        for item in items:
            if "tests/unit/" in str(item.fspath):
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)
    elif config.getoption("--e2e"):
        selected = []
        deselected = []
        for item in items:
            if "tests/e2e/" in str(item.fspath):
                selected.append(item)
            else:
                deselected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

