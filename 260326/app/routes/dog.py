from flask import Blueprint,render_template, request
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

router = Blueprint("Dog", __name__, url_prefix="/dog", template_folder=BASE_DIR.parent/"templates")

with open(BASE_DIR.parent/"models/dog_model.pkl", "rb") as f:
    model = pickle.load(f)


# ============ Data 준비 ============
dog_names = ["Dachshund", "Samoyed"]

# 닥스훈트 길이 & 높이
dach_length = [77, 78, 85, 83, 73, 77, 73, 80]
dach_height = [25, 28, 29, 30, 21, 22, 17, 35]

# 사모예드 길이 & 높이
samo_length = [75, 77, 86, 86, 79, 83, 83, 88]
samo_height = [56, 57, 50, 53, 60, 53, 49, 61]

dog_lengh = np.array(dach_length + samo_length)
dog_height = np.array(dach_height + samo_height)

dog_data = np.column_stack((dog_lengh, dog_height)) # [lengh, height] 구조로 합침



# ============ 그래프 표시 함수 ============
def model_predict_plot(x):
    dog_labels = model.predict(dog_data)
    result_label = model.predict(x)

    # 닥스훈트 (0)
    plt.scatter(
        dog_data[dog_labels == 0][:, 0],
        dog_data[dog_labels == 0][:, 1],
        c="green",
        label="Dachshund",
        alpha=0.6
    )

    # 사모예드 (1)
    plt.scatter(
        dog_data[dog_labels == 1][:, 0],
        dog_data[dog_labels == 1][:, 1],
        c="blue",
        label="Samoyed",
        alpha=0.6
    )

    plt.scatter(x[:, 0], x[:, 1], c="red")

    for i in range(len(x)):
        plt.annotate(
            "Your Dog",           # 표시할 라벨
            xy=(x[i, 0], x[i, 1]), # 점 좌표
            xytext=(5, 5),         # 점 기준으로 글자 위치 이동
            textcoords="offset points", # offset 기준 단위
            fontsize=9,
            color="red",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",  # 둥근 박스 + 여백
                fc="white",                # 배경색 (facecolor)
                ec="red",                  # 테두리 색 (edgecolor)
                lw=1                       # 테두리 두께
        )
        )

    plt.xlim((70, 90))
    plt.ylim((10, 70))
    
    plt.xticks([])
    plt.yticks([])

    plt.legend()
    
    plt.savefig(BASE_DIR.parent/"static/results/result.png")
    plt.close()

    return result_label



@router.route("/", methods=["GET", "POST"])
def dog_kmeans():
    prediction = None
    image_name= None
    graph_path = "result.png"
    
    if request.method == "POST":
        try:
            length = float(request.form["length"])
            height = float(request.form["height"])

            features = np.array([[length, height]])

            label = model_predict_plot(features)
            prediction = dog_names[label[0]]
            image_name = f"{dog_names[label[0]]}.jpg"

            graph_path = "result.png"
        except Exception as e:
            prediction = f"입력 오류 : {e}"
        
    return render_template("dog_kmeans.html", prediction=prediction, image_name=image_name, graph_path=graph_path)