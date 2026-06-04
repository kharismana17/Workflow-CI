import os
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("StressPrediction_Basic")

mlflow.autolog()

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset_preprocessing",
    "stress_clean.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop("Stress_Level", axis=1)
y = df["Stress_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")