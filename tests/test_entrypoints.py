from app import app as app_entrypoint
from wsgi import app as wsgi_entrypoint


def test_entrypoints_expose_flask_app(main_api_module):
    assert app_entrypoint is main_api_module.app
    assert wsgi_entrypoint is main_api_module.app
