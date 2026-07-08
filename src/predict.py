import joblib
import pandas as pd


def main():
    model = joblib.load("models/random_forest.pkl")
    encoder = joblib.load("models/label_encoder.pkl")

    df = pd.read_csv("data/raw/Crop_recommendation.csv")

    X = df.drop("label", axis=1)

    sample = X.iloc[[0]]

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    crop = encoder.inverse_transform([prediction])[0]

    probability_df = pd.DataFrame({
        "Crop": encoder.classes_,
        "Probability": probability
    }).sort_values(
        by="Probability",
        ascending=False
    )

    print("Sample Features:")
    print(sample)

    print("\nRecommended Crop:")
    print(crop)

    print("\nTop 5 Predicted Crops:")

    print(probability_df.head())
    

if __name__ == "__main__":
    main()