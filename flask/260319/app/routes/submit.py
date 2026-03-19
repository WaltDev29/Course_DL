from flask import Blueprint, request, redirect, render_template

router = Blueprint("Submit", __name__, url_prefix="/submit")

@router.route("", methods=["GET", "POST"])
def submit_page():
    if request.method == "POST":
        data = request.form
        name = data["username"]
        age = data["age"]
        return {
            "success":True,
            "message": "POST ok",
            "data": {
                "name": name,
                "age": age
            }
        }, 201
    
    elif request.method == "GET":
        return redirect("https://naver.com")