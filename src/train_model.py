import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import joblib
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import seaborn as sns

# Import model libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Import sklearn base classes for wrapper classes
from sklearn.base import BaseEstimator, ClassifierMixin

# Import configuration utilities
from utils.config import (
    load_config,
    get_data_paths,
    get_model_config,
    get_training_config,
    get_paths,
    create_directories
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Wrapper classes defined at module level for pickling
class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model
        self.classes_ = None
        
    def fit(self, X, y):
        # Clean feature names for XGBoost
        X_clean = X.copy()
        
        # Handle object columns
        object_cols = X_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            X_clean[col] = X_clean[col].astype('category').cat.codes
        
        # Clean column names
        new_columns = {}
        for col in X_clean.columns:
            new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            if new_col != col:
                new_columns[col] = new_col
        
        if new_columns:
            X_clean = X_clean.rename(columns=new_columns)
        
        # Set classes_ attribute
        self.classes_ = np.unique(y)
        
        # Fit the underlying model
        self.xgb_model.fit(X_clean, y)
        return self
        
    def predict(self, X):
        # Clean feature names for prediction
        X_clean = X.copy()
        
        # Handle object columns
        object_cols = X_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            X_clean[col] = X_clean[col].astype('category').cat.codes
        
        # Clean column names
        new_columns = {}
        for col in X_clean.columns:
            new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            if new_col != col:
                new_columns[col] = new_col
        
        if new_columns:
            X_clean = X_clean.rename(columns=new_columns)
        
        # Make predictions with the underlying model
        return self.xgb_model.predict(X_clean)
        
    def predict_proba(self, X):
        # Clean feature names for prediction
        X_clean = X.copy()
        
        # Handle object columns
        object_cols = X_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            X_clean[col] = X_clean[col].astype('category').cat.codes
        
        # Clean column names
        new_columns = {}
        for col in X_clean.columns:
            new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            if new_col != col:
                new_columns[col] = new_col
        
        if new_columns:
            X_clean = X_clean.rename(columns=new_columns)
        
        # Make predictions with the underlying model
        return self.xgb_model.predict_proba(X_clean)


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, lgbm_model):
        self.lgbm_model = lgbm_model
        self.classes_ = None
        
    def fit(self, X, y):
        # Convert object dtypes for LightGBM
        X_clean = X.copy()
        object_cols = X_clean.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            X_clean[col] = X_clean[col].astype('category').cat.codes
        
        # Set classes_ attribute
        self.classes_ = np.unique(y)
        
        # Fit the model
        self.lgbm_model.fit(X_clean, y)
        return self
    
    def predict(self, X):
        X_clean = X.copy()
        object_cols = X_clean.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            X_clean[col] = X_clean[col].astype('category').cat.codes
        
        return self.lgbm_model.predict(X_clean)
    
    def predict_proba(self, X):
        X_clean = X.copy()
        object_cols = X_clean.select_dtypes(include=['object']).columns
        
        for col in object_cols:
            X_clean[col] = X_clean[col].astype('category').cat.codes
        
        return self.lgbm_model.predict_proba(X_clean)


def load_data():
    """Load preprocessed data."""
    # Get data paths
    data_paths = get_data_paths()
    processed_paths = data_paths.get("processed", {})
    
    # Load the data
    logger.info(f"Loading training data from {processed_paths.get('train')}")
    train_df = pd.read_parquet(processed_paths.get("train"))
    
    logger.info(f"Loading validation data from {processed_paths.get('validation')}")
    val_df = pd.read_parquet(processed_paths.get("validation"))
    
    logger.info(f"Loading test data from {processed_paths.get('test')}")
    test_df = pd.read_parquet(processed_paths.get("test"))
    
    # Split features and target
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    
    X_val = val_df.drop("target", axis=1)
    y_val = val_df["target"]
    
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_model(model_name, tuned=False):
    """Get model instance with configured parameters."""
    # Get model configuration
    config = get_model_config(model_name, tuned)
    
    # Initialize the appropriate model
    if model_name == "xgboost":
        return XGBClassifier(**config)
    elif model_name == "lightgbm":
        return LGBMClassifier(**config)
    elif model_name == "catboost":
        # Create a copy of config to avoid modifying the original
        catboost_config = config.copy()
        # Remove cat_features if it exists - we'll handle them separately
        cat_features = catboost_config.pop('cat_features', None)
        return CatBoostClassifier(**catboost_config), cat_features
    elif model_name == "randomforest":
        return RandomForestClassifier(**config)
    elif model_name == "stacking":
        # Handle stacking model configuration
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Get base models from config
        base_model_names = config.pop('base_models', ['xgboost', 'lightgbm', 'catboost'])
        base_models = []
        
        # Extract random state before creating models
        random_state = config.pop('random_state', 42)
        
        # Initialize base models
        for base_model_name in base_model_names:
            if base_model_name == "catboost":
                # For CatBoost, we only need the model, not the cat_features
                base_model_result = get_model(base_model_name, tuned)
                if isinstance(base_model_result, tuple):
                    base_model = base_model_result[0]
                else:
                    base_model = base_model_result
            elif base_model_name == "xgboost":
                # Wrap XGBoost model to handle feature names automatically
                xgb_model = XGBClassifier(**get_model_config(base_model_name, tuned))
                base_model = XGBoostWrapper(xgb_model)
            elif base_model_name == "lightgbm":
                # Wrap LightGBM model for consistent handling
                lgbm_model = LGBMClassifier(**get_model_config(base_model_name, tuned))
                base_model = LGBMWrapper(lgbm_model)
            else:
                base_model = get_model(base_model_name, tuned)
            
            base_models.append((base_model_name, base_model))
        
        # Get meta-model
        meta_model_name = config.pop('meta_model', 'logistic_regression')
        if meta_model_name == 'logistic_regression':
            meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            # Add support for other meta-models as needed
            meta_model = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # Create stacking model
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            **config
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def train_model(model, X_train, y_train, X_val, y_val, cat_features=None):
    """Train a model and evaluate on validation set."""
    # Train the model
    logger.info(f"Training {type(model).__name__}")
    
    # Convert object dtype columns to numeric for all models
    # Check if there are any object dtype columns
    object_cols = X_train.select_dtypes(include=['object']).columns
    
    # Special handling for XGBoost - clean feature names
    if isinstance(model, XGBClassifier):
        logger.info("Preparing data for XGBoost")
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        
        # Convert object columns to category codes
        if len(object_cols) > 0:
            logger.info(f"Converting {len(object_cols)} object columns to numeric for XGBoost")
            for col in object_cols:
                X_train_xgb[col] = X_train_xgb[col].astype('category').cat.codes
                X_val_xgb[col] = X_val_xgb[col].astype('category').cat.codes
        
        # Clean feature names - remove special characters
        new_columns = {}
        for col in X_train_xgb.columns:
            # Replace any special characters that XGBoost doesn't like
            new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            if new_col != col:
                new_columns[col] = new_col
        
        if new_columns:
            logger.info(f"Renaming {len(new_columns)} columns for XGBoost compatibility")
            X_train_xgb = X_train_xgb.rename(columns=new_columns)
            X_val_xgb = X_val_xgb.rename(columns=new_columns)
        
        # Train the model
        model.fit(X_train_xgb, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val_xgb)
        y_pred_proba = model.predict_proba(X_val_xgb)[:, 1]
    
    # If we have object columns, we need to handle them according to the model type
    elif len(object_cols) > 0:
        logger.info(f"Found {len(object_cols)} object columns: {list(object_cols)}")
        
        # Special handling for CatBoost to properly set categorical features
        if isinstance(model, CatBoostClassifier):
            # If cat_features not provided, try to detect categorical columns
            if cat_features is None:
                # Identify potential categorical features (object or category dtype)
                cat_features_indices = []
                for i, col in enumerate(X_train.columns):
                    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category' or 'education_' in col or 'marital_status_' in col:
                        cat_features_indices.append(i)
                        
                if cat_features_indices:
                    logger.info(f"Detected {len(cat_features_indices)} categorical features for CatBoost")
                    # Convert categorical features to string to ensure proper handling
                    X_train_copy = X_train.copy()
                    X_val_copy = X_val.copy()
                    
                    # Train with cat_features parameter
                    model.fit(X_train_copy, y_train, cat_features=cat_features_indices)
                    
                    # Evaluate
                    y_pred = model.predict(X_val_copy)
                    y_pred_proba = model.predict_proba(X_val_copy)[:, 1]
                else:
                    # No categorical features detected
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
            else:
                # Use provided cat_features
                logger.info(f"Using {len(cat_features)} specified categorical features for CatBoost")
                model.fit(X_train, y_train, cat_features=cat_features)
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # For all other models, convert object columns to category codes
        else:
            logger.info(f"Converting {len(object_cols)} object columns to numeric for {type(model).__name__}")
            X_train_converted = X_train.copy()
            X_val_converted = X_val.copy()
            
            for col in object_cols:
                # Convert to category then to codes
                X_train_converted[col] = X_train_converted[col].astype('category').cat.codes
                X_val_converted[col] = X_val_converted[col].astype('category').cat.codes
            
            # Train the model with converted data
            model.fit(X_train_converted, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val_converted)
            y_pred_proba = model.predict_proba(X_val_converted)[:, 1]
    
    else:
        # No object columns, train normally
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba)
    }
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_val, y_pred))
    
    # Store mapping information for XGBoost if columns were renamed
    if isinstance(model, XGBClassifier) and 'new_columns' in locals() and new_columns:
        # Attach the column mapping to the model for later use
        model.column_mapping = new_columns
    
    return model, metrics


def save_model(model, model_name, tuned=False):
    """Save trained model to disk."""
    # Get paths
    paths = get_paths()
    models_dir = paths.get("models", "models/")
    
    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate filename
    tuned_str = "tuned" if tuned else "baseline"
    filename = f"{model_name}_{tuned_str}.pkl"
    filepath = os.path.join(models_dir, filename)
    
    # Save model
    logger.info(f"Saving model to {filepath}")
    joblib.dump(model, filepath)
    
    return filepath


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance."""
    # Get paths
    paths = get_paths()
    figures_dir = paths.get("figures", "figures/")
    
    # Create directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check if model supports feature importance
    has_feature_importance = hasattr(model, "feature_importances_")
    
    # Special handling for LightGBM
    if isinstance(model, LGBMClassifier) and hasattr(model, "booster_"):
        # LightGBM has a different way to access feature importance
        importances = model.booster_.feature_importance(importance_type='gain')
        has_feature_importance = True
    # Regular feature importance
    elif has_feature_importance:
        importances = model.feature_importances_
    else:
        logger.warning(f"Model {model_name} doesn't have feature_importances_ attribute")
        return
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    plt.bar(range(min(20, len(indices))), importances[indices[:20]], align="center")
    plt.xticks(range(min(20, len(indices))), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.title(f"Top 20 Feature Importances - {model_name}")
    plt.tight_layout()
    
    # Save figure
    filename = f"{model_name}_feature_importance.png"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath)
    logger.info(f"Feature importance plot saved to {filepath}")


def plot_confusion_matrix(model, X_val, y_val, model_name):
    """Plot confusion matrix."""
    # Get paths
    paths = get_paths()
    figures_dir = paths.get("figures", "figures/")
    
    # Create directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Special handling for XGBoost
    if isinstance(model, XGBClassifier):
        X_val_xgb = X_val.copy()
        
        # Convert object columns to category codes
        object_cols = X_val_xgb.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                X_val_xgb[col] = X_val_xgb[col].astype('category').cat.codes
        
        # Apply column renaming if the model has a stored mapping
        if hasattr(model, 'column_mapping') and model.column_mapping:
            X_val_xgb = X_val_xgb.rename(columns=model.column_mapping)
        else:
            # If no mapping stored, apply the same cleaning logic
            new_columns = {}
            for col in X_val_xgb.columns:
                new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                if new_col != col:
                    new_columns[col] = new_col
            
            if new_columns:
                X_val_xgb = X_val_xgb.rename(columns=new_columns)
        
        # Get predictions
        y_pred = model.predict(X_val_xgb)
    
    # Check if there are any object dtype columns for other models
    elif len(X_val.select_dtypes(include=['object']).columns) > 0:
        object_cols = X_val.select_dtypes(include=['object']).columns
        
        # Special handling for CatBoost
        if isinstance(model, CatBoostClassifier):
            X_val_copy = X_val.copy()
            y_pred = model.predict(X_val_copy)
        
        # For all other models, convert object columns to category codes
        else:
            X_val_converted = X_val.copy()
            
            for col in object_cols:
                # Convert to category then to codes
                X_val_converted[col] = X_val_converted[col].astype('category').cat.codes
            
            # Get predictions
            y_pred = model.predict(X_val_converted)
    else:
        # No object columns, predict normally
        y_pred = model.predict(X_val)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save figure
    filename = f"{model_name}_confusion_matrix.png"
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath)
    logger.info(f"Confusion matrix plot saved to {filepath}")


def evaluate_model_on_test(model, X_test, y_test, model_name):
    """Evaluate model on test set."""
    logger.info(f"Evaluating {model_name} on test set")
    
    # Special handling for XGBoost
    if isinstance(model, XGBClassifier):
        X_test_xgb = X_test.copy()
        
        # Convert object columns to category codes
        object_cols = X_test_xgb.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                X_test_xgb[col] = X_test_xgb[col].astype('category').cat.codes
        
        # Apply column renaming if the model has a stored mapping
        if hasattr(model, 'column_mapping') and model.column_mapping:
            X_test_xgb = X_test_xgb.rename(columns=model.column_mapping)
        else:
            # If no mapping stored, apply the same cleaning logic
            new_columns = {}
            for col in X_test_xgb.columns:
                new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                if new_col != col:
                    new_columns[col] = new_col
            
            if new_columns:
                X_test_xgb = X_test_xgb.rename(columns=new_columns)
        
        # Get predictions
        y_pred = model.predict(X_test_xgb)
        y_pred_proba = model.predict_proba(X_test_xgb)[:, 1]
    
    # Check if there are any object dtype columns for other models
    elif len(X_test.select_dtypes(include=['object']).columns) > 0:
        object_cols = X_test.select_dtypes(include=['object']).columns
        
        # Special handling for CatBoost
        if isinstance(model, CatBoostClassifier):
            # Identify potential categorical features (object or category dtype)
            cat_features_indices = []
            for i, col in enumerate(X_test.columns):
                if X_test[col].dtype == 'object' or X_test[col].dtype.name == 'category' or 'education_' in col or 'marital_status_' in col:
                    cat_features_indices.append(i)
                    
            # Use copy to avoid modifying original data
            X_test_copy = X_test.copy()
            y_pred = model.predict(X_test_copy)
            y_pred_proba = model.predict_proba(X_test_copy)[:, 1]
        
        # For all other models, convert object columns to category codes
        else:
            X_test_converted = X_test.copy()
            
            for col in object_cols:
                # Convert to category then to codes
                X_test_converted[col] = X_test_converted[col].astype('category').cat.codes
            
            # Get predictions
            y_pred = model.predict(X_test_converted)
            y_pred_proba = model.predict_proba(X_test_converted)[:, 1]
    
    else:
        # No object columns, predict normally
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Print classification report
    logger.info("\nTest Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return metrics


def save_results(model_name, val_metrics, test_metrics, tuned=False):
    """Save results to CSV file."""
    # Get paths
    paths = get_paths()
    results_dir = paths.get("results", "results/")
    
    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Combine metrics
    all_metrics = {**{f"val_{k}": v for k, v in val_metrics.items()}, 
                  **test_metrics}
    
    # Create DataFrame
    tuned_str = "tuned" if tuned else "baseline"
    results_df = pd.DataFrame({
        "model": [f"{model_name}_{tuned_str}"],
        **{k: [v] for k, v in all_metrics.items()}
    })
    
    # Generate filename
    filename = "model_results.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Check if file exists
    if os.path.exists(filepath):
        # Append to existing file
        existing_df = pd.read_csv(filepath)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    logger.info(f"Results saved to {filepath}")


def main():
    """Main function to train and evaluate models."""
    # Create necessary directories
    create_directories()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Get list of feature names
    feature_names = X_train.columns.tolist()
    
    # Get model types from config
    config = load_config()
    
    model_types = list(config.get("models", {}).keys())
    
    # Loop through models
    for model_name in model_types:
        logger.info(f"Processing model: {model_name}")
        
        # Process baseline model
        logger.info(f"Training baseline {model_name} model")
        
        # Handle CatBoost specially
        cat_features = None
        if model_name == "catboost":
            model_result = get_model(model_name, tuned=False)
            if isinstance(model_result, tuple):
                model, cat_features = model_result
            else:
                model = model_result
        else:
            model = get_model(model_name, tuned=False)
        
        # Train the model
        model, val_metrics = train_model(model, X_train, y_train, X_val, y_val, cat_features)
        
        # Save baseline model
        model_path = save_model(model, model_name, tuned=False)
        
        # Plot feature importance and confusion matrix
        plot_feature_importance(model, feature_names, f"{model_name}_baseline")
        plot_confusion_matrix(model, X_val, y_val, f"{model_name}_baseline")
        
        # Evaluate on test set
        test_metrics = evaluate_model_on_test(model, X_test, y_test, f"{model_name}_baseline")
        
        # Save results
        save_results(model_name, val_metrics, test_metrics, tuned=False)
        
        # Process tuned model
        logger.info(f"Training tuned {model_name} model")
        
        # Handle CatBoost specially
        cat_features = None
        if model_name == "catboost":
            model_result = get_model(model_name, tuned=True)
            if isinstance(model_result, tuple):
                tuned_model, cat_features = model_result
            else:
                tuned_model = model_result
        else:
            tuned_model = get_model(model_name, tuned=True)
        
        # Train the model
        tuned_model, tuned_val_metrics = train_model(tuned_model, X_train, y_train, X_val, y_val, cat_features)
        
        # Save tuned model
        tuned_model_path = save_model(tuned_model, model_name, tuned=True)
        
        # Plot feature importance and confusion matrix
        plot_feature_importance(tuned_model, feature_names, f"{model_name}_tuned")
        plot_confusion_matrix(tuned_model, X_val, y_val, f"{model_name}_tuned")
        
        # Evaluate on test set
        tuned_test_metrics = evaluate_model_on_test(tuned_model, X_test, y_test, f"{model_name}_tuned")
        
        # Save results
        save_results(model_name, tuned_val_metrics, tuned_test_metrics, tuned=True)
        
        logger.info(f"Completed processing {model_name} models")
    
    logger.info("All models trained and evaluated successfully")

if __name__ == "__main__":
    main()