from flask import Flask, render_template

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template("index.html")

    from .routes import users
    app.register_blueprint(users.router)

    from .routes import submit
    app.register_blueprint(submit.router)

    return app