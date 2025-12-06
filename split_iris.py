import pandas as pd

# Load the iris dataset
df = pd.read_csv('iris.csv')

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"\nFirst few rows:")
print(df.head())

# Shuffle the dataset randomly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index (75% for training)
split_index = int(len(df_shuffled) * 0.75)

# Split into training and testing
train_df = df_shuffled[:split_index]
test_df = df_shuffled[split_index:]

print(f"\nTraining set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Save the splits to CSV files
train_df.to_csv('iris_train.csv', index=False)
test_df.to_csv('iris_test.csv', index=False)

print("\nSaved training data to: iris_train.csv")
print("Saved testing data to: iris_test.csv")