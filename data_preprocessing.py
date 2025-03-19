import numpy as np
import pandas as pd # dataframes
import matplotlib.pyplot as plt # General visualisations
import matplotlib.ticker as mtick # Axis visuals
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def load_process_data(building_file):
    
    # Load data
    df = pd.read_csv(building_file,index_col = 'building_id')
    
    
    print(f"Structure data has {len(df)} rows and {df.shape[1]} columns")

    #drop rows with missing data
    df.dropna(inplace=True)

    # Convert certain columns to object types (they are categorical)
    df = df.astype({'district_id': 'object', 'vdcmun_id': 'object', 'ward_id': 'object'})

    # Drop rows with missing data
    df.dropna(inplace = True)

    # Drop irrelevant columns
    cols_to_drop = ['count_floors_post_eq', 'height_ft_post_eq', 'condition_post_eq', 'technical_solution_proposed']
    #df.drop(cols_to_drop, axis=1, inplace=True)

    # Dataframe shape before preprocessing
    print(f"Before preprocessing there were {len(df)} rows and {df.shape[1]} columns")


    # Identify categorical columns
    object_features = ['district_id', 'vdcmun_id', 'ward_id'] #categorical id's
    new_categorical = df.drop(object_features, axis = 1).select_dtypes(include=['object']).drop('damage_grade', axis = 1).columns
    numerical_features = df.select_dtypes(np.number).columns

    # Convert damage grade to just number
    df['damage_grade'] = df['damage_grade'].str.extract('(\\d+)').astype(int)

    # Convert to object Transformer
    df[object_features] = df[object_features].astype('category')

    #standardization 
    mean = df[numerical_features].mean()
    std_dev = df[numerical_features].std()
    df[numerical_features] = (df[numerical_features] - mean) / std_dev

    #minmax scaling
    X_min = df[numerical_features].min()
    X_max = df[numerical_features].max()
    df[numerical_features] = (df[numerical_features] - X_min) / (X_max - X_min)

    # Categorical Transformer
    df = pd.get_dummies(df, columns=object_features + list(new_categorical), drop_first=True)

    #remove near zero variance features
    selector = VarianceThreshold(threshold=.005)
    df = pd.DataFrame(selector.fit_transform(df), columns=df.columns[selector.get_support()])

    def plotDistribution(df):
        # Calculate actual damage category frequencies
        damage_counts = df['damage_grade'].value_counts().sort_index()

        # Generate bar plot
        plt.figure(figsize=(8, 5))
        plt.bar(damage_counts.index, damage_counts.values, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel("Building Damage Category")
        plt.ylabel("Frequency")
        plt.title("Distribution of Building Damage Categories")
        plt.xticks(damage_counts.index)  # Ensure correct x-axis labels
        plt.show()

    #plotDistribution(df)

    # Split data into training and testing (damage grade is y value)
    x_train, x_test, y_train, y_test = train_test_split(df.drop('damage_grade', axis = 1), df['damage_grade'],
                                                        test_size = 0.2, random_state = 1989, stratify = df['damage_grade'], shuffle=True)

    # Convert all to float (for svd)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Print x train shape before processing
    print(f"After preprocessing there are {len(x_train)} rows and {x_train.shape[1]} columns")

    return x_train, x_test, y_train, y_test


   

if __name__ == "__main__":
    load_process_data("csv_building_structure.csv")
    