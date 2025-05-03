import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def load_process_data(building_file):
    """
    Load and preprocess building structure data for earthquake damage classification.

    Parameters:
    - building_file: Path to CSV file containing building data

    Returns:
    - x_train: Processed training features
    - x_test: Processed test features
    - y_train: Training labels
    - y_test: Test labels
    """
    
    # Load CSV and set building_id as index
    df = pd.read_csv(building_file, index_col='building_id')
    print(f"Structure data has {len(df)} rows and {df.shape[1]} columns")

    # Drop rows with any missing data
    df.dropna(inplace=True)

    # Convert ID columns to categorical types
    df = df.astype({'district_id': 'object', 'vdcmun_id': 'object', 'ward_id': 'object'})

    # Drop unnecessary post-earthquake or proposal-related columns
    cols_to_drop = ['count_floors_post_eq', 'height_ft_post_eq', 'condition_post_eq', 'technical_solution_proposed']
    # Uncomment to drop if needed:
    # df.drop(cols_to_drop, axis=1, inplace=True)

    print(f"Before preprocessing: {len(df)} rows and {df.shape[1]} columns")

    # Identify feature types
    object_features = ['district_id', 'vdcmun_id', 'ward_id']
    new_categorical = df.drop(object_features, axis=1).select_dtypes(include=['object']).drop('damage_grade', axis=1).columns
    numerical_features = df.select_dtypes(np.number).columns

    # Convert damage grade strings (e.g. 'Grade 5') to integers
    df['damage_grade'] = df['damage_grade'].str.extract(r'(\d+)').astype(int)

    # Convert ID columns to categorical
    df[object_features] = df[object_features].astype('category')

    # Standardize numeric features
    mean = df[numerical_features].mean()
    std_dev = df[numerical_features].std()
    df[numerical_features] = (df[numerical_features] - mean) / std_dev

    # Min-max scaling
    X_min = df[numerical_features].min()
    X_max = df[numerical_features].max()
    df[numerical_features] = (df[numerical_features] - X_min) / (X_max - X_min)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=object_features.tolist() + list(new_categorical), drop_first=True)

    # Remove near-zero variance features
    selector = VarianceThreshold(threshold=0.005)
    df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])

    # Optional: Plot distribution of damage grades
    # plot_distribution(df)

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop('damage_grade', axis=1), df['damage_grade'],
        test_size=0.2, random_state=1989, stratify=df['damage_grade'], shuffle=True
    )

    # Convert to float32 for compatibility with SVD and ML models
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print(f"After preprocessing: {len(x_train)} rows and {x_train.shape[1]} columns")

    return x_train, x_test, y_train, y_test


def plot_distribution(df):
    """
    Plot the frequency distribution of damage categories.

    Parameters:
    - df: DataFrame with 'damage_grade' column
    """
    damage_counts = df['damage_grade'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(damage_counts.index, damage_counts.values, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Building Damage Category")
    plt.ylabel("Frequency")
    plt.title("Distribution of Building Damage Categories")
    plt.xticks(damage_counts.index)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    load_process_data("csv_building_structure.csv")
