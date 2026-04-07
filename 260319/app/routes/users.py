from flask import Blueprint, request, abort, render_template

router = Blueprint("Users", __name__, url_prefix="/users")

# ============ GET ============
# Path Parameter
@router.route("/<name>")
def get_user(name:str):
    return render_template("user_home.html", name=name), 200


# Query Parameter
@router.route("")
def get_users():
    name = request.args.get("name", "Guest")
    return {
        "success":True,
        "message": "Query ok",
        "data": name
    }, 200



# ============ POST ============
@router.route("", methods=["POST"])
def create_user():
    data = request.get_json()
    if not "name" in data:
        abort(400)

    return {
        "success":True,
        "message": "POST ok",
        "data": data
    }, 201