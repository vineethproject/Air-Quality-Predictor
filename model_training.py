import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load the air quality dataset from a CSV file.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"File {filepath} not found. Creating synthetic data for demonstration...")
        return create_synthetic_data()

def create_synthetic_data():
    """
    Create synthetic air quality data for demonstration purposes.
    
    Returns:
    pd.DataFrame: Synthetic dataset
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic pollutant data
    data = {
        'PM2.5': np.random.normal(50, 20, n_samples),
        'PM10': np.random.normal(80, 30, n_samples),
        'NO': np.random.normal(30, 15, n_samples),
        'NO2': np.random.normal(40, 20, n_samples),
        'NH3': np.random.normal(25, 10, n_samples),
        'CO': np.random.normal(1.5, 0.8, n_samples),
        'SO2': np.random.normal(20, 12, n_samples),
        'O3': np.random.normal(60, 25, n_samples)
    }
    
    # Ensure no negative values
    for key in data:
        data[key] = np.abs(data[key])
    
    df = pd.DataFrame(data)
    print("Synthetic data created for demonstration.")
    return df

def preprocess_data(data):
    """
    Preprocess the air quality data by handling missing values and calculating AQI.
    
    Parameters:
    data (pd.DataFrame): Raw air quality data
    
    Returns:
    pd.DataFrame: Preprocessed data with AQI column
    """
    print("Starting data preprocessing...")
    
    # Handle missing values using forward fill
    pollutant_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NH3', 'CO', 'SO2', 'O3']
    
    # Check which columns exist in the data
    existing_columns = [col for col in pollutant_columns if col in data.columns]
    
    if not existing_columns:
        print("Warning: No expected pollutant columns found. Using available numeric columns.")
        existing_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values
    for col in existing_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
    
    # Calculate AQI (simplified version - using maximum normalized value)
    # Normalize each pollutant to 0-100 scale and take the maximum
    normalized_data = data[existing_columns].copy()
    
    # Simple normalization based on typical AQI breakpoints
    normalization_factors = {
        'PM2.5': 150,  # AQI 200 at 150 µg/m³
        'PM10': 250,   # AQI 200 at 250 µg/m³
        'NO': 100,     # Approximate scale
        'NO2': 100,    # Approximate scale
        'NH3': 100,    # Approximate scale
        'CO': 10,      # Approximate scale
        'SO2': 100,    # Approximate scale
        'O3': 200      # AQI 200 at 200 µg/m³
    }
    
    for col in existing_columns:
        if col in normalization_factors:
            normalized_data[col] = (data[col] / normalization_factors[col]) * 100
        else:
            # Generic normalization for unknown columns
            normalized_data[col] = (data[col] / data[col].max()) * 100
    
    # Calculate AQI as the maximum of normalized pollutant values
    data['AQI'] = normalized_data[existing_columns].max(axis=1)
    
    # Ensure AQI is within reasonable bounds
    data['AQI'] = data['AQI'].clip(0, 500)
    
    print(f"Preprocessing completed. Final shape: {data.shape}")
    print(f"AQI range: {data['AQI'].min():.2f} - {data['AQI'].max():.2f}")
    
    return data, existing_columns

def train_and_evaluate_models(X, y):
    """
    Train and evaluate Random Forest and XGBoost models.
    
    Parameters:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target variable (AQI)
    
    Returns:
    tuple: Best model and its name
    """
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training models...")
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_model_name = None
    best_r2 = -np.inf
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'model': model, 'rmse': rmse, 'r2': r2}
        
        print(f"{name} Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        # Check if this is the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} (R² = {best_r2:.4f})")
    
    return best_model, best_model_name, results

def save_model(model, model_name, feature_columns):
    """
    Save the best model and feature information.
    
    Parameters:
    model: Trained model object
    model_name (str): Name of the model
    feature_columns (list): List of feature column names
    """
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/aqi_predictor_model.pkl'
    joblib.dump(model, model_path)
    
    # Save feature columns for consistency
    feature_path = 'models/feature_columns.pkl'
    joblib.dump(feature_columns, feature_path)
    
    print(f"\nModel saved successfully!")
    print(f"Model: {model_path}")
    print(f"Features: {feature_path}")
    print(f"Best model was: {model_name}")

def visualize_results(data, results):
    """
    Create visualizations of the data and model results.
    
    Parameters:
    data (pd.DataFrame): Preprocessed data
    results (dict): Model results dictionary
    """
    plt.figure(figsize=(15, 10))
    
    # AQI distribution
    plt.subplot(2, 3, 1)
    plt.hist(data['AQI'], bins=30, alpha=0.7, color='skyblue')
    plt.title('AQI Distribution')
    plt.xlabel('AQI')
    plt.ylabel('Frequency')
    
    # Model performance comparison
    plt.subplot(2, 3, 2)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    plt.bar(model_names, r2_scores, color=['lightcoral', 'lightgreen'])
    plt.title('Model Performance Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    
    # RMSE comparison
    plt.subplot(2, 3, 3)
    rmse_scores = [results[name]['rmse'] for name in model_names]
    plt.bar(model_names, rmse_scores, color=['lightsalmon', 'lightblue'])
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # Correlation matrix
    plt.subplot(2, 3, (4, 6))
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlation_matrix = data[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('models/model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to 'models/model_analysis.png'")

def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    print("Starting Air Quality Index Prediction Model Training...")
    print("=" * 60)
    
    # Load data
    data_path = 'data/city_day.csv'
    data = load_data(data_path)
    
    # Preprocess data
    processed_data, feature_columns = preprocess_data(data)
    
    # Prepare features and target
    X = processed_data[feature_columns]
    y = processed_data['AQI']
    
    print(f"\nFeature columns: {feature_columns}")
    print(f"Target variable: AQI")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Train and evaluate models
    best_model, best_model_name, results = train_and_evaluate_models(X, y)
    
    # Save the best model
    save_model(best_model, best_model_name, feature_columns)
    
    # Create visualizations
    visualize_results(processed_data, results)
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("You can now run the Streamlit app using: streamlit run app.py")

if __name__ == "__main__":
    main()
