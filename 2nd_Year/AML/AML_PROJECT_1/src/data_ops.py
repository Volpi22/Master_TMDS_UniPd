############################### IMPORTS ############################################

from ucimlrepo import fetch_ucirepo 
import pandas as pd
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.transformation import BoxCoxTransformer
from sklearn.preprocessing import StandardScaler

############################# COLUMN DEFINITIONS ########################################

# Define categorical columns and their data types
categorical_cols = {
        'cp': 'category',
        'restecg': 'category',
        'thal': 'category',
        'slope': 'category',
        'ca': 'category',
    }

# Define numerical columns and their data types
numerical_cols = {
        'trestbps': 'float64',
        'chol': 'float64',
        'thalach': 'float64',
        'age': 'float64',
    }

# Define special columns that require unique handling
special_cols = {
        'oldpeak': 'float64',
}

binary_cols = {
        'sex': 'int64',
        'fbs': 'int64',
        'exang': 'int64',
}

############################# DATA LOADING ########################################

def load_heart_disease_data():
    """
    Load the heart disease dataset from UCI ML repository.
    
    Returns:
    --------
    tuple
        - X: Feature matrix (DataFrame)
        - y: Target variable (Series, binary: 0=no disease, 1=disease present)
    """
    # Fetch dataset from UCI repository (ID 45 is heart disease)
    heart_disease = fetch_ucirepo(id=45) 
    X = heart_disease.data.features
    y = heart_disease.data.targets 
    # Convert multi-class target (0-4) to binary (0=no disease, 1-4=disease present)
    y = (y['num'] > 0).astype(int)

    return X, y

############################# DATA PREPROCESSING ########################################

def preprocess_heart_disease_data(X, y):
    """
    Preprocess the heart disease dataset.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataset with:
        - Missing values removed from 'ca' and 'thal' columns
        - Categorical variables converted to category dtype
        - Target variable included as 'num' column
    """
    # Combine features and target
    dataset = pd.concat([X, y], axis=1)
    # Print number of NA values and columns containing them
    na_counts = dataset.isna().sum()
    na_columns = na_counts[na_counts > 0]
    print(f"Number of NA values per column:")
    print(na_columns)
    print(f"Total NA values: {na_columns.sum()}")
    # Remove rows with missing values in 'ca' and 'thal' columns
    dataset = dataset.dropna(subset=['ca', 'thal'])
    print("NA values removed" + "\n")

    # Convert categorical columns to category dtype
    for col, dtype in categorical_cols.items():
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(dtype)
    return dataset

############################## DATA TRANSFORMATION ########################################

def data_transformation(train_data, test_data):
    """
    Apply power transformations to numerical features to normalize distributions.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training dataset
    test_data : pd.DataFrame
        Test dataset
    
    Returns:
    --------
    tuple
        - train_data: Transformed training dataset
        - test_data: Transformed test dataset (using training parameters)
    
    Note:
    -----
    - BoxCox transformation used for strictly positive data (requires all values > 0)
    - Yeo-Johnson transformation used for data with zero or negative values
    - Transformations fitted on training data only to prevent data leakage
    """
    for col in numerical_cols.keys():
        # Check if all training values are positive for BoxCox
        if train_data[col].min() > 0:
            boxcox = BoxCoxTransformer(variables=[col])
            boxcox.fit(train_data)
            train_data = boxcox.transform(train_data)
            test_data = boxcox.transform(test_data)
        else:
            # Use Yeo-Johnson for non-positive data
            yoejohnson = YeoJohnsonTransformer(variables=[col])
            yoejohnson.fit(train_data)
            train_data = yoejohnson.transform(train_data)
            test_data = yoejohnson.transform(test_data)
            
    return train_data, test_data

############################## DATA STANDARDIZATION ########################################

def data_standardization(train_data, test_data):
    """
    Standardize numerical features using StandardScaler (z-score normalization).
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training dataset
    test_data : pd.DataFrame
        Test dataset
    
    Returns:
    --------
    tuple
        - train_data: Standardized training dataset
        - test_data: Standardized test dataset (using training statistics)
    
    Note:
    -----
    - Scaler is fit only on training data to prevent data leakage
    - Transforms features to have mean=0 and std=1
    """
    scaler = StandardScaler()
    continuous_cols = list(numerical_cols.keys())
    # Fit on training data and transform both sets
    train_data[continuous_cols] = scaler.fit_transform(train_data[continuous_cols])
    test_data[continuous_cols] = scaler.transform(test_data[continuous_cols])

    return train_data, test_data

############################### OUTLIER IDENTIFICATION ########################################

def find_outliers(dataset):
    """
    Identify outliers in numerical features using IQR method.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    tuple
        - outliers_count: pd.Series with count of outliers for each numerical feature
        - outliers_df: pd.DataFrame containing all observations that have at least one outlier value
    
    Note:
    -----
    Outliers defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    where IQR = Q3 - Q1 (Interquartile Range)
    """
    continuous_features = list(numerical_cols.keys()) + list(special_cols.keys())
    # Calculate quartiles
    Q1 = dataset[continuous_features].quantile(0.25)
    Q3 = dataset[continuous_features].quantile(0.75)
    IQR = Q3 - Q1
    
    # Create mask for outliers
    outliers_mask = (dataset[continuous_features] < (Q1 - 1.5 * IQR)) | (dataset[continuous_features] > (Q3 + 1.5 * IQR))
    
    # Count values outside the IQR bounds
    outliers_count_specified = outliers_mask.sum()
    
    # Get rows with at least one outlier
    rows_with_outliers = outliers_mask.any(axis=1)
    outliers_df = dataset[rows_with_outliers].copy()
    
    # Add a column indicating which features have outliers for each observation
    outlier_features = []
    for idx in outliers_df.index:
        outlier_cols = [col for col in continuous_features if outliers_mask.loc[idx, col]]
        outlier_features.append(', '.join(outlier_cols) if outlier_cols else 'None')
    outliers_df['outlier_features'] = outlier_features

    return outliers_count_specified, outliers_df

############################### OUTLIER REMOVAL ########################################

def remove_outliers(dataset, columns=None):
    """
    Remove outliers from dataset using IQR method.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset to remove outliers from
    columns : list, optional
        List of column names to check for outliers. If None, uses numerical_cols keys
    
    Returns:
    --------
    pd.DataFrame
        Dataset with outliers removed
    
    Note:
    -----
    Outliers are defined as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    Rows containing outliers in any of the specified columns are removed.
    """
    if columns is None:
        columns = list(numerical_cols.keys())
    
    # Keep only columns that exist in the dataset
    columns = [col for col in columns if col in dataset.columns]
    
    dataset_clean = dataset.copy()
    
    for col in columns:
        Q1 = dataset_clean[col].quantile(0.25)
        Q3 = dataset_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove rows with outliers
        dataset_clean = dataset_clean[(dataset_clean[col] >= lower_bound) & (dataset_clean[col] <= upper_bound)]
    
    print(f"Original dataset shape: {dataset.shape}")
    print(f"Dataset after removing outliers: {dataset_clean.shape}")
    print(f"Rows removed: {dataset.shape[0] - dataset_clean.shape[0]}")
    
    return dataset_clean

if __name__ == "__main__":
    pass