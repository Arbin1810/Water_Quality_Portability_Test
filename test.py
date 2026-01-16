import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def compare_models():
    """
    Compare different ML models for water potability prediction
    Returns: DataFrame with model comparison results
    """
    # Load and preprocess data
    file_path = r'water_quality_potability.csv'
    water_df = pd.read_csv(file_path)
    
    # Data cleaning
    water_df = water_df.drop_duplicates()
    water_df = water_df.dropna()
    
    # Remove outliers from Solids column
    Q1 = water_df.Solids.quantile(0.25)
    Q3 = water_df.Solids.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    water_df = water_df[(water_df.Solids > lower_limit) & (water_df.Solids < upper_limit)]
    
    # Prepare features and target
    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    X = water_df[feature_names]
    y = water_df['Potability']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    # Initialize models with best hyperparameters (pre-determined through grid search)
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=17, metric='manhattan', weights='distance'),
        'SVM': SVC(C=1, gamma='scale', kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    
    # Train and evaluate each model
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': round(accuracy * 100, 2),
            'Precision': round(precision * 100, 2),
            'Recall': round(recall * 100, 2),
            'F1-Score': round(f1 * 100, 2)
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model based on Recall (most important for water safety)
    results_df['Best_Model'] = results_df['Recall'] == results_df['Recall'].max()
    
    return results_df, scaler, models

def train_single_model(model_name):
    """
    Train a single ML model for water potability prediction
    
    Args:
        model_name: Name of the model to train ('KNN', 'SVM', or 'Logistic Regression')
    
    Returns:
        model: The trained model
        scaler: Fitted StandardScaler object
        metrics: Dictionary containing trained model and its metrics
    """
    # Load and preprocess data
    file_path = r'water_quality_potability.csv'
    water_df = pd.read_csv(file_path)
    
    # Data cleaning
    water_df = water_df.drop_duplicates()
    water_df = water_df.dropna()
    
    # Remove outliers from Solids column
    Q1 = water_df.Solids.quantile(0.25)
    Q3 = water_df.Solids.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    water_df = water_df[(water_df.Solids > lower_limit) & (water_df.Solids < upper_limit)]
    
    # Prepare features and target
    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    X = water_df[feature_names]
    y = water_df['Potability']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    
    metrics = {}
    
    if model_name == 'KNN':
        print(f"Training {model_name}...")
        model = KNeighborsClassifier(n_neighbors=17, metric='manhattan', weights='distance')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics[model_name] = {
            'model': model,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2),
            'best_params': {'n_neighbors': 17, 'metric': 'manhattan', 'weights': 'distance'}
        }
    
    elif model_name == 'SVM':
        print(f"Training {model_name}...")
        model = SVC(C=1, gamma='scale', kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics[model_name] = {
            'model': model,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2),
            'best_params': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
        }
    
    elif model_name == 'Logistic Regression':
        print(f"Training {model_name}...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics[model_name] = {
            'model': model,
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2),
            'best_params': {'max_iter': 1000}
        }
    else:
        print(f"Unknown model name: {model_name}")
        return None, None, None
    
    print(f"{model_name} training completed!")
    return model, scaler, metrics

# If run directly, execute model comparison
if __name__ == "__main__":
    results_df, scaler, models = compare_models()
    print("Model Comparison Results:")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\nBest Model(s) based on Recall Score:")
    print(results_df[results_df['Best_Model']][['Model', 'Recall']].to_string(index=False))