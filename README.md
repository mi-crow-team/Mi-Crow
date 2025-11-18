![CI](https://github.com/AdamKaniasty/Inzynierka/actions/workflows/tests.yml/badge.svg?branch=main)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://adamkaniasty.github.io/Inzynierka/)

## Running Tests

The project uses pytest for testing. Tests are organized into unit tests and end-to-end tests.

### Running All Tests

```bash
pytest
```

### Running Specific Test Suites

Run only unit tests:
```bash
pytest --unit -q
```

Run only end-to-end tests:
```bash
pytest --e2e -q
```

You can also use pytest markers:
```bash
pytest -m unit -q
pytest -m e2e -q
```

Or specify the test directory directly:
```bash
pytest tests/unit -q
pytest tests/e2e -q
```

### Test Coverage

The test suite is configured to require at least 85% code coverage. Coverage reports are generated in both terminal and XML formats.