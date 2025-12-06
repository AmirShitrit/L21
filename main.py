import pandas as pd
from data_utils import split_data

# Load the iris dataset
df = pd.read_csv('iris.csv')

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"\nFirst few rows:")
print(df.head())

# Split into training and testing using the imported function
train_df, test_df = split_data(df, train_ratio=0.75, random_state=42)

print(f"\nTraining set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

