import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR/"models"
MODEL_FILE = MODEL_DIR/"iris_model.pkl"

iris = datasets.load_iris()
x = iris.data
y = iris.target

train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.2, random_state=42)



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data, train_label)



Path(MODEL_DIR).mkdir(exist_ok=True)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(knn, f)

print("모델 저장 완료")