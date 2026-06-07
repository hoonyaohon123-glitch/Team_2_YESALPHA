import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

from FeatureEngineering import engineer_features, load_cleaned_data

def run_data_pipeline():
    print(f"Loading data")
    df = load_cleaned_data()
    print(f"Engineering features")
    engineered_df = engineer_features(df)
    print("Dropping unnecessary columns and preparing target variable...")

    X = engineered_df.drop(columns=['Target_Activity', 'Session ID']) # both will not be used in training
    y = engineered_df['Target_Activity'] # target variable for classification
    return X, y

# Once the data is loaded and features are engineered, we proceed to split and scale the data for model training in the next part.

def split_data(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    # Splitting the data into training and testing sets with stratification to maintain class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test



def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Scaling numerical features using StandardScaler. Categorical features are left unchanged for this step.

    scaler = StandardScaler()
    cols_to_scale = [c for c in X_train.columns if not c.startswith(('HVAC', 'Time')) and X_train[c].nunique() > 2]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    print(f"Scaled features: {cols_to_scale}")
    print("Feature scaling complete.")
    return X_train_scaled, X_test_scaled

# ML_load_set_data.py

def get_data_for_training():
    # having a singular function gives us all the variables i need for model training immediately.
    X, y = run_data_pipeline()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    get_data_for_training()
    print("Data preparation complete. Ready for model training and evaluation.")
    