
import pandas as pd
import glob
import os
os.makedirs("data/processed", exist_ok=True)
# Step 1: get all files
files = glob.glob("data/raw/*.csv")

print("Files found:", files)

df_list = []

for file in files:
    df = pd.read_csv(file)

    # convert column names to lowercase
    df.columns = df.columns.str.lower()

    print(f"\nColumns in {file}:", df.columns.tolist())

    # keep only required columns
    cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    df = df[[c for c in cols if c in df.columns]]

    # add source column (important for research paper)
    df['source'] = file

    df_list.append(df)

# combine all data
data = pd.concat(df_list, ignore_index=True)

# clean data
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

print("\nFinal shape:", data.shape)

# save combined dataset
data.to_csv("data/processed/combined_data.csv", index=False)

print("\n✅ Combined dataset saved!")