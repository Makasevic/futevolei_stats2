from __future__ import annotations

import sys

import pytest


if __name__ == "__main__":
    default_args = ["tests", "--ignore-glob=tests/pytest-cache-files-*"]
    raise SystemExit(pytest.main(sys.argv[1:] or default_args))
