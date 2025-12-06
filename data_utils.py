import pandas as pd


def split_data(df, train_ratio=0.75, random_state=42):
    """
    Split a dataframe into training and testing sets.

    Args:
        df: pandas DataFrame to split
        train_ratio: proportion of data for training (default 0.75)
        random_state: random seed for reproducibility (default 42)

    Returns:
        tuple: (train_df, test_df)
    """
    # Shuffle the dataset randomly
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split index
    split_index = int(len(df_shuffled) * train_ratio)

    # Split into training and testing
    train_df = df_shuffled[:split_index]
    test_df = df_shuffled[split_index:]

    return train_df, test_df