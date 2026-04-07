from flask import Blueprint, render_template

router = Blueprint("Iris", __name__, url_prefix="/iris")

@router.route("")
def iris_home():
    return render_template("iris_home.html")