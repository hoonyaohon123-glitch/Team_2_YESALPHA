import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV 
from imblearn.over_sampling import SMOTE

import json
import os # for saving model and params with dynamic paths
from sklearn.pipeline import Pipeline
import joblib

from ML_load_set_data import get_data_for_training

X_train_scaled, X_test_scaled, y_train, y_test = get_data_for_training()

# Baseline Model Training and Evaluation

def evaluate_baseline_models(X_train, X_test, y_train, y_test):
    print("Evaluating baseline models: Logistic Regression, Random Forest, and XGBoost.")
    # Auto loops through a set of baseline models, trains them, and evaluates their performance using classification reports and macro F1-scores.
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42)
        # All the models have a random_state set for reproducibility, and Logistic Regression has an increased max_iter to ensure convergence.
    }

    best_model_name = ""
    best_macro_f1 = 0.0
    class_labels = ["Low Activity", "Moderate Activity", "High Activity"] # These are the text labels for the target

    for name, model in models.items():
        print(f"Training {name}:")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\n{name} Report:")
        print(classification_report(y_test, y_pred, target_names=class_labels))
        print(f"Macro F1-Score (main evaluator): {macro_f1:.4f}\n")


        if macro_f1 > best_macro_f1: # if the score is the best we've seen, we update our tracking variables
            best_macro_f1 = macro_f1
            best_model_name = name

    print("\n")
    print(f"EXPLORATION COMPLETE")
    print(f"The best model is **{best_model_name}** with a Macro F1 of {best_macro_f1:.4f}")


def run_training_pipeline_tuned(X_train, X_test, y_train, y_test):
    print("Running the training pipeline with tuned hyperparameters...")
    print("Applying class balancing technique using SMOTE")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("SMOTE balancing applied to training data.")
    
    # Training loop for RandomisedSearchCV with all the same models as before, but with tuned hyperparameters.
    # Defining all our models and their hyperparameter grids for tuning.
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=42),
            "params": {
                'C': [0.01, 0.1, 1, 10, 100, 200],
                'solver': ['lbfgs', 'saga'],
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6, 9, 12],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1],
                'gamma': [0, 0.1, 0.2],
            }
        }
    }

    best_model_name = ""
    best_macro_f1 = 0.0
    class_labels = ["Low Activity", "Moderate Activity", "High Activity"]
    
    for name, model_info in models.items():
        print(f"Training {name} with hyperparameter tuning:")
        random_search = RandomizedSearchCV(
            estimator=model_info["model"],
            param_distributions=model_info["params"],
            n_iter=20,  # Number of random combinations to try
            scoring='f1_macro',  # Optimize for macro F1-score
            cv=5,  # 5-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        random_search.fit(X_train_balanced, y_train_balanced)
        best_estimator = random_search.best_estimator_
        best_params = random_search.best_params_
        print(f"Best hyperparameters for {name}: {best_params}")

        y_pred = best_estimator.predict(X_test)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\n{name} Report (Tuned):")
        print(classification_report(y_test, y_pred, target_names=class_labels))
        print(f"Macro F1-Score (main evaluator): {macro_f1:.4f}\n")

        # Saving the best paras in a JSON file AND model itself.
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_model')
        BEST_PARAMS_DIR = os.path.join(PROJECT_ROOT, 'saved_model')


        if not os.path.exists(SAVED_MODELS_DIR):
            os.makedirs(SAVED_MODELS_DIR)
        if not os.path.exists(BEST_PARAMS_DIR):
            os.makedirs(BEST_PARAMS_DIR)
        model_filename = f"{name.replace(' ', '_')}_best_model.pkl"
        params_filename = f"{name.replace(' ', '_')}_best_params.json"
        joblib.dump(best_estimator, os.path.join(SAVED_MODELS_DIR, model_filename))
        with open(os.path.join(BEST_PARAMS_DIR, params_filename), 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Saved best model for {name} to {os.path.join(SAVED_MODELS_DIR, model_filename)}")
        print(f"Saved best hyperparameters for {name} to {os.path.join(BEST_PARAMS_DIR, params_filename)}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_name = name

    print("\n")
    print(f"TUNED EXPLORATION COMPLETE")
    print(f"The best tuned model is **{best_model_name}** with a Macro F1 of {best_macro_f1:.4f}")
    

if __name__ == "__main__":
    print("BASELINE models & Classification Report:")
    evaluate_baseline_models(X_train_scaled, X_test_scaled, y_train, y_test)
    print("\n\n")
    print("TUNED models & Classification Report:")
    run_training_pipeline_tuned(X_train_scaled, X_test_scaled, y_train, y_test)

    # Show best params and features
