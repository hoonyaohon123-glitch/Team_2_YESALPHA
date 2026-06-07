#!/bin/bash

# Exit immediately if any command fails
set -e

echo "======================================================="
echo "  Starting Gas Monitoring Machine Learning Pipeline    "
echo "======================================================="

# ---------------------------------------------------------
# Phase 1: Data Cleaning & Exploratory Data Analysis
# ---------------------------------------------------------
echo "--> [1/2] Executing Data Cleaning Pipeline (EDAPython.py)..."
#Execute EDAPython.py file
python EDAPython.py
echo "--> Data cleaning complete. Cleaned CSV generated."
echo ""

# ---------------------------------------------------------
# Phase 2: Feature Engineering & Model Training
# ---------------------------------------------------------
# Note: ML_train_model.py automatically handles the feature engineering in memory before training the models.
echo "--> [2/2] Executing Model Training Pipeline (ML_train_model.py)..."
#Execute ML_train_model.py file
python ML_train_model.py

echo "======================================================="
echo "  Pipeline Execution Complete!                         "
echo "  Check the 'saved_model' directory for outputs.       "
echo "======================================================="