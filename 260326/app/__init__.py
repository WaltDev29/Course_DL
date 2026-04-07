from flask import Flask

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return "Home"

    from .routes import iris
    app.register_blueprint(iris.router)

    from .routes import dog
    app.register_blueprint(dog.router)

    return app