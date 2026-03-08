import pandas as pd

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Safety
    required = ["age", "bmi", "smoker", "children"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["smoker"] = data["smoker"].astype(str).str.lower().str.strip()

    # BMI Category
    data["BMI_Category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    # Age Group
    data["Age_Group"] = pd.cut(
        data["age"],
        bins=[0, 30, 50, 100],
        labels=["Young", "Middle", "Senior"]
    )

    # Interaction features
    data["Smoker_Risk_Index"] = (data["smoker"] == "yes").astype(int) * data["bmi"]
    data["Family_Load"] = data["children"] * data["age"]

    # Lifestyle Risk Score
    data["Lifestyle_Risk_Score"] = (
        0.4 * (data["bmi"] / 50) +
        0.4 * (data["smoker"] == "yes").astype(int) +
        0.2 * (data["age"] / 100)
    )

    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data


# Optional: allow running this file directly to generate final_data.csv
if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/processed/combined_data.csv")
    df2 = feature_engineering(df)
    df2.to_csv("data/processed/final_data.csv", index=False)
    print("✅ Saved: data/processed/final_data.csv", df2.shape)
    # import pandas as pd
# import os

# os.makedirs("data/processed", exist_ok=True)

# # 1) Load combined dataset
# data = pd.read_csv("data/processed/combined_data.csv")

# print("Loaded:", data.shape)

# # 2) BMI Category
# data["BMI_Category"] = pd.cut(
#     data["bmi"],
#     bins=[0, 18.5, 25, 30, 100],
#     labels=["Underweight", "Normal", "Overweight", "Obese"]
# )
# data["smoker"] = data["smoker"].astype(str).str.lower().str.strip()

# # 3) Age Group
# data["Age_Group"] = pd.cut(
#     data["age"],
#     bins=[0, 30, 50, 100],
#     labels=["Young", "Middle", "Senior"]
# )

# # 4) Interaction features
# data["Smoker_Risk_Index"] = (data["smoker"] == "yes").astype(int) * data["bmi"]
# data["Family_Load"] = data["children"] * data["age"]

# # 5) Lifestyle Risk Score (simple weighted score)
# data["Lifestyle_Risk_Score"] = (
#     0.4 * (data["bmi"] / 50) +
#     0.4 * (data["smoker"] == "yes").astype(int) +
#     0.2 * (data["age"] / 100)
# )

# # 6) Clean
# data.dropna(inplace=True)
# data.drop_duplicates(inplace=True)

# print("After feature engineering:", data.shape)
# print("Columns:", data.columns.tolist())

# # 7) Save final dataset
# data.to_csv("data/processed/final_data.csv", index=False)
# print("✅ Saved: data/processed/final_data.csv")