from flask import Blueprint

router = Blueprint("test", __name__, url_prefix="/")

@router.route("")
def home():
    return "Hello Flask!"


@router.route("test")
def test():
    return "TESTTESTTEST"