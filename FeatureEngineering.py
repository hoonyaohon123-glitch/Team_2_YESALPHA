"""
feature_engineering.py
----------------------
This script takes the cleaned gas monitoring dataset and engineers 
advanced temporal, rolling, and aggregated features to improve 
machine learning model accuracy.

Output: data/model_ready_gas_monitoring.csv
"""

import pandas as pd
import numpy as np
import os
import sys

def load_cleaned_data(relative_path: str = 'data/cleaned_gas_monitoring.csv') -> pd.DataFrame:
    """
    Loads the cleaned dataset using dynamic absolute paths to prevent VS Code and Docker errors.
    Explicitly looks inside the 'data' subfolder.
    """
    # Automatically detect the exact folder where this Python script is saved
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Attach the 'data/...' path to it
    file_path = os.path.join(current_dir, relative_path)
    
    print(f"Attempting to load data from:\n  -> {file_path}")
    
    # Failsafe: Check if the file actually exists before trying to read it
    if not os.path.exists(file_path):
        print(f"\n[ERROR] File not found!")
        print(f"Please ensure 'cleaned_gas_monitoring.csv' is actually saved inside the 'data' folder.")
        print(f"Looked exactly here: {file_path}")
        sys.exit(1) # Stops the script from freezing
        
    return pd.read_csv(file_path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering transformations to the dataframe.
    Assumes data is chronologically ordered within each Session ID.
    """
    print("\nEngineering features...")
    df = df.copy()

    # -------------------------------------------------------------------------
    # 1. Temporal Dynamics (Rate of Change)
    # -------------------------------------------------------------------------
    print(" -> Calculating temporal dynamics...")
    df['CO2_Infrared_Diff'] = df.groupby('Session ID')['CO2_InfraredSensor'].diff().fillna(0)
    df['Temp_Diff'] = df.groupby('Session ID')['Temperature'].diff().fillna(0)
    df['Humidity_Diff'] = df.groupby('Session ID')['Humidity'].diff().fillna(0)

    # -------------------------------------------------------------------------
    # 2. Rolling Window Statistics (Smoothing & Volatility)
    # -------------------------------------------------------------------------
    print(" -> Calculating rolling window statistics...")
    df['Temp_Volatility'] = df.groupby('Session ID')['Temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )
    df['CO2_Volatility'] = df.groupby('Session ID')['CO2_InfraredSensor'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )

    # -------------------------------------------------------------------------
    # 3. Sensor Fusion (Aggregated Meta-Features)
    # -------------------------------------------------------------------------
    print(" -> Fusing metal oxide sensors...")
    metal_oxide_cols = [
        'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 
        'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4'
    ]
    df['MetalOxide_Mean'] = df[metal_oxide_cols].mean(axis=1)
    df['MetalOxide_Max'] = df[metal_oxide_cols].max(axis=1)

    # -------------------------------------------------------------------------
    # 4. Session-Level Baselines
    # -------------------------------------------------------------------------
    print(" -> Calculating session baselines...")
    df['Temp_Centered'] = df['Temperature'] - df.groupby('Session ID')['Temperature'].transform('mean')
    df['CO2_Centered'] = df['CO2_InfraredSensor'] - df.groupby('Session ID')['CO2_InfraredSensor'].transform('mean')

    # -------------------------------------------------------------------------
    # 5. Categorical Encoding (Model Readiness)
    # -------------------------------------------------------------------------
    print(" -> Encoding categorical variables...")
    
    # Target Variable Ordinal Encoding
    activity_map = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df['Target_Activity'] = df['Activity Level'].map(activity_map)
    
    # Ambient Light Ordinal Encoding
    light_map = {'very_dim': 0, 'dim': 1, 'normal': 2, 'bright': 3, 'very_bright': 4}
    df['Ambient_Light_Encoded'] = df['Ambient Light Level'].map(light_map).fillna(2)

    # One-Hot Encoding for Nominal Categories
    df = pd.get_dummies(df, columns=['HVAC Operation Mode', 'Time of Day'], drop_first=True)

    # Drop the original raw text columns that are now encoded
    df = df.drop(columns=['Activity Level', 'Ambient Light Level'])

    return df

def main():
    print("=== Starting Feature Engineering Pipeline ===")
    
    # 1. Load the data (now automatically looking in the data/ folder)
    df = load_cleaned_data('data/cleaned_gas_monitoring.csv')
    
    # 2. Apply feature engineering
    df_features = engineer_features(df)
    
    # 3. Summary of new dataset
    print("\n=== Feature Engineering Complete ===")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {df_features.shape}")
    print(f"Total Features: {len(df_features.columns)}")
    
    # 4. Save to CSV in the EXACT same 'data' folder
    output_filename = 'data/model_ready_gas_monitoring.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_filename)
    
    # Ensure the data directory exists before saving, just in case
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_features.to_csv(output_path, index=False)
    print(f"\nModel-ready dataset saved successfully to:\n  -> {output_path}")

if __name__ == "__main__":
    main()