import pandas as pd
df = pd.read_csv("data_processed/seizeit2_features.csv")

print(df.shape) # Expected: ~4000-8000 rows hone chahiye 31 runs ki wajah se
print(df['patient_id'].value_counts()) # 3 patients
print(df['Label'].value_counts()) # Seizure windows ~100-300 hone chahiye
print(df.isna().sum()) # NaN check