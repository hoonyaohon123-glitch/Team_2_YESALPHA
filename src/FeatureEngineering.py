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
    # """Loads the cleaned dataset using dynamic absolute paths.
    # Resolving the absolute path prevents 'FileNotFound' errors caused by 
    # differences in how VS Code or Docker sets the system working directory."""
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(current_dir, relative_path)
    
    # print(f"Attempting to load data from:\n  -> {file_path}")
    
    # if not os.path.exists(file_path):
    #     print(f"\n[ERROR] File not found! Looked exactly here: {file_path}")
    #     sys.exit(1)
    current_dir = os.path.dirname(os.path.abspath(__file__)) 

    # Construct the absolute path based on the script's location
    file_path = os.path.join(current_dir, '..', 'data', 'cleaned_gas_monitoring.csv')

    df = pd.read_csv(file_path)
    # Edited: Used Absolute Path because I cannot find the relative path that works to be used in the ML_draft Notebook.
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies advanced feature engineering transformations. 
    Such as temporal dynamics, rolling statistics, and domain-specific environmental metrics"""
    print("\nEngineering features...")
    df = df.copy()

    # -------------------------------------------------------------------------
    # 1. Temporal Dynamics
    # -------------------------------------------------------------------------
    # Captures the immediate change from one second to the next.
    print(" -> Calculating temporal dynamics...")

    # Calculate how much CO2 changed compared to the exact previous reading.
    df['CO2_Infrared_Diff'] = df.groupby('Session ID')['CO2_InfraredSensor'].diff().fillna(0)

    # Calculate the step-by-step change in Temperature.
    df['Temp_Diff'] = df.groupby('Session ID')['Temperature'].diff().fillna(0)

    # Calculate the step-by-step change in Humidity.
    df['Humidity_Diff'] = df.groupby('Session ID')['Humidity'].diff().fillna(0)

    # -------------------------------------------------------------------------
    # 2. Rolling Window Statistics
    # -------------------------------------------------------------------------
    # Calculate averages and volatility
    print(" -> Calculating rolling window statistics...")

    # Calculate the standard deviation of Temperature over the last 5 readings to detect sudden fluctuations.
    df['Temp_Volatility'] = df.groupby('Session ID')['Temperature'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )

    # Measure the jumps in CO2 readings over the last 5 time steps.
    df['CO2_Volatility'] = df.groupby('Session ID')['CO2_InfraredSensor'].transform(
        lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
    )
    
    # Calculate the average CO2 level over the last 10 readings to make a baseline.
    df['CO2_Rolling_Mean'] = df.groupby('Session ID')['CO2_InfraredSensor'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

    # -------------------------------------------------------------------------
    # 3. Sensor Fusion (Means and Maxes)
    # -------------------------------------------------------------------------
    # Combine multiple sensors into a single metric
    print(" -> Fusing metal oxide sensors...")

    # Group the 4 redundant Metal Oxide sensor columns together into a list.
    metal_oxide_cols = [
        'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 
        'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4'
    ]

    # Combine the 4 sensors into a single average value to represent general air quality.
    df['MetalOxide_Mean'] = df[metal_oxide_cols].mean(axis=1)

    # Find the highest reading among the 4 sensors for this specific moment.
    df['MetalOxide_Max'] = df[metal_oxide_cols].max(axis=1)

    # -------------------------------------------------------------------------
    # 4. Session-Level Baselines
    # -------------------------------------------------------------------------
    # Measures the current reading difference from the normal average.
    print(" -> Calculating session baselines...")

    # Measure how far the current Temperature is from the average Temperature of this specific session.
    df['Temp_Centered'] = df['Temperature'] - df.groupby('Session ID')['Temperature'].transform('mean')

    # Measure how far the current CO2 level is from this specific session's overall average.
    df['CO2_Centered'] = df['CO2_InfraredSensor'] - df.groupby('Session ID')['CO2_InfraredSensor'].transform('mean')

    # -------------------------------------------------------------------------
    # 5. Domain-Specific Features (Air Quality & Environmental Comfort)
    # -------------------------------------------------------------------------
    # Applies actual scientific formulas to the raw numbers.
    print(" -> Calculating Environmental Comfort, VOC Burden, and Discrepancy Flags...")
    
    # A. Environmental Comfort Index (Thom's Discomfort Index)
    # Apply a meteorological formula using Temperature and Humidity to estimate human physical comfort. 
    df['Env_Comfort_Index'] = df['Temperature'] - 0.55 * (1 - (df['Humidity'] / 100)) * (df['Temperature'] - 14.5)

    # B. Total VOC Air Burden (Sum of all units)
    # Add up the raw values of all 4 Metal Oxide sensors to get the absolute total amount of gas in the air.
    df['Total_VOC_Burden'] = df[metal_oxide_cols].sum(axis=1)
    # Rolling VOC Burden 
    # Sum up the total gas burden over the last 10 readings to see how much gas has accumulated.
    df['VOC_Accumulation_10'] = df.groupby('Session ID')['Total_VOC_Burden'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum()
    )

    # C. CO2 Sensor Discrepancy Flag
    # Flag instantaneous spikes.
    df['CO2_Discrepancy_Flag'] = (abs(df['CO2_InfraredSensor'] - df['CO2_Rolling_Mean']) > 30).astype(int)
    
    # Delete the 'CO2_Rolling_Mean' column since it was only temporary.
    df = df.drop(columns=['CO2_Rolling_Mean'])

    # -------------------------------------------------------------------------
    # 6. Text-to-Numeric Translation (Ordinal Mapping & Dummy Variables)
    # -------------------------------------------------------------------------
    # Translates English text into mathematical vectors.
    print(" -> Encoding categorical variables...")
    
    # Convert the text labels for 'Activity Level' into mathematical numbers (0, 1, 2).
    activity_map = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df['Target_Activity'] = df['Activity Level'].map(activity_map)
    
    # Convert text-based light levels into an ordered numerical scale, defaulting to 2 if missing.
    light_map = {'very_dim': 0, 'dim': 1, 'moderate': 2, 'bright': 3, 'very_bright': 4}
    df['Ambient_Light_Encoded'] = df['Ambient Light Level'].map(light_map).fillna(2)

    # Convert nominal text categories into binary
    df = pd.get_dummies(df, columns=['HVAC Operation Mode', 'Time of Day'], drop_first=True)

    # Delete the original raw text columns
    df = df.drop(columns=['Activity Level', 'Ambient Light Level'])

    return df