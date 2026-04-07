from flask import Blueprint,render_template, request
import numpy as np
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent

router = Blueprint("Iris", __name__, url_prefix="/iris", template_folder=BASE_DIR.parent/"templates")

with open(BASE_DIR.parent/"models/iris_model.pkl", "rb") as f:
    model = pickle.load(f)


iris_names = ["setosa", "versicolor", "virginica"]


@router.route("/", methods=["GET", "POST"])
def iris_knn():
    prediction = None
    image_name= None

    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            pred = model.predict(features)[0]
            prediction = iris_names[pred]

            image_name = f"{prediction}.jpg"

        except Exception as e:
            prediction = f"입력 오류 : {e}"
        
    return render_template("iris_knn.html", prediction=prediction, image_name=image_name)

