import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)


def main():
    df = pd.read_csv("data/raw/Crop_recommendation.csv")

    X = df.drop("label", axis=1)
    y = df["label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=12,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1 Score",
            "OOB Score"
        ],
        "Value": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average="weighted"),
            recall_score(y_test, y_pred, average="weighted"),
            f1_score(y_test, y_pred, average="weighted"),
            model.oob_score_
        ]
    })

    metrics.to_csv(
        "outputs/tables/evaluation_metrics.csv",
        index=False
    )

    report = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            output_dict=True
        )
    ).transpose()

    report.to_csv(
        "outputs/tables/classification_report.csv"
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "outputs/tables/feature_importance.csv",
        index=False
    )

    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(encoder, "models/label_encoder.pkl")

    print(metrics)


if __name__ == "__main__":
    main()