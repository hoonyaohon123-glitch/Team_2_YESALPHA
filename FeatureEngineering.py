"""
feature_engineering.py
----------------------
This script takes the cleaned gas monitoring dataset and engineers 
advanced temporal, rolling, aggregated, and domain-specific features 
to improve machine learning model accuracy.
"""

import pandas as pd
import numpy as np
import os
import sys

def load_cleaned_data(relative_path: str = 'data/cleaned_gas_monitoring.csv') -> pd.DataFrame:
    """Loads the cleaned dataset using dynamic absolute paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, relative_path)
    
    print(f"Attempting to load data from:\n  -> {file_path}")
    
    if not os.path.exists(file_path):
        print(f"\n[ERROR] File not found! Looked exactly here: {file_path}")
        sys.exit(1)
        
    return pd.read_csv(file_path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies advanced feature engineering transformations."""
    print("\nEngineering features...")
    df = df.copy()

    # -------------------------------------------------------------------------
    # 1. Temporal Dynamics
    # -------------------------------------------------------------------------
    print(" -> Calculating temporal dynamics...")
    df['CO2_Infrared_Diff'] = df.groupby('Session ID')['CO2_InfraredSensor'].diff().fillna(0)
    df['Temp_Diff'] = df.groupby('Session ID')['Temperature'].diff().fillna(0)
    df['Humidity_Diff'] = df.groupby('Session ID')['Humidity'].diff().fillna(0)

    # -------------------------------------------------------------------------
    # 2. Rolling Window Statistics
    # -------------------------------------------------------------------------
    print(" -> Calculating rolling window statistics...")
    df['Temp_Volatility'] = df.groupby('Session ID')['Temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )
    df['CO2_Volatility'] = df.groupby('Session ID')['CO2_InfraredSensor'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )
    
    # Calculate a rolling mean for CO2 (Used later for the Discrepancy Flag)
    df['CO2_Rolling_Mean'] = df.groupby('Session ID')['CO2_InfraredSensor'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

    # -------------------------------------------------------------------------
    # 3. Sensor Fusion (Means and Maxes)
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
    # 5. Domain-Specific Features (Air Quality & Environmental Comfort)
    # -------------------------------------------------------------------------
    print(" -> Calculating Environmental Comfort, VOC Burden, and Discrepancy Flags...")
    
    # A. Environmental Comfort Index (Thom's Discomfort Index)
    df['Env_Comfort_Index'] = df['Temperature'] - 0.55 * (1 - (df['Humidity'] / 100)) * (df['Temperature'] - 14.5)

    # B. Total VOC Air Burden (Sum of all units)
    df['Total_VOC_Burden'] = df[metal_oxide_cols].sum(axis=1)
    
    # Rolling VOC Burden (How much gas has accumulated over the last 10 readings?)
    df['VOC_Accumulation_10'] = df.groupby('Session ID')['Total_VOC_Burden'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )

    # C. CO2 Sensor Discrepancy Flag (Binary 0 or 1)
    df['CO2_Discrepancy_Flag'] = (abs(df['CO2_InfraredSensor'] - df['CO2_Rolling_Mean']) > 30).astype(int)
    
    # Drop the temporary rolling mean column to keep the dataset clean
    df = df.drop(columns=['CO2_Rolling_Mean'])

    # -------------------------------------------------------------------------
    # 6. Text-to-Numeric Translation (Ordinal Mapping & Dummy Variables)
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

    # Drop the original raw text columns
    df = df.drop(columns=['Activity Level', 'Ambient Light Level'])

    return df