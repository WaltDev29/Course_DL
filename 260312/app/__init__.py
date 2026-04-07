from flask import Flask
from . import router

def create_app():
    app = Flask(__name__)

    app.register_blueprint(router.router)

    return app