from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    """Temporary app factory that reuses the legacy Flask app.

    This keeps the new entrypoints stable while the route migration out of
    ``main_api.py`` is still in progress.
    """

    from main_api import app as legacy_app

    return legacy_app
