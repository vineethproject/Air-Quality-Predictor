# Air Quality Index (AQI) Prediction System 🌍

A machine learning-based system for predicting Air Quality Index (AQI) using historical weather and pollution data, featuring a user-friendly Streamlit web application for real-time predictions.

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict Air Quality Index (AQI) values based on various pollutant concentrations. The system includes data preprocessing, model training with multiple algorithms, model evaluation, and a web-based interface for making predictions.

### Key Features

- **Machine Learning Models**: Implements both Random Forest and XGBoost algorithms
- **Automated Model Selection**: Chooses the best performing model based on R² score
- **Interactive Web Interface**: Streamlit-based dashboard for easy predictions
- **Data Visualization**: Comprehensive charts and graphs for data analysis
- **Real-time Predictions**: Instant AQI predictions with health impact assessments
- **Synthetic Data Support**: Generates demo data when real dataset is not available

## 🏗️ Project Structure

`
AirQualityPredictor/
│
├── data/                          # Data directory (add your datasets here)
│   └── city_day.csv              # Expected dataset (user-provided)
│
├── models/                       # Saved models and artifacts
│   ├── aqi_predictor_model.pkl   # Trained model (generated)
│   ├── feature_columns.pkl       # Feature column names (generated)
│   └── model_analysis.png        # Model performance visualizations (generated)
│
├── app.py                        # Streamlit web application
├── model_training.py             # Model training and evaluation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
`

## 🚀 Quick Start

### 1. Installation

First, clone or download this project to your local machine, then install the required dependencies:

`ash
# Navigate to the project directory
cd AirQualityPredictor

# Install required packages
pip install -r requirements.txt
`

### 2. Train the Model

Run the model training script to prepare your machine learning model:

`ash
python model_training.py
`

**What this script does:**
- Loads data from data/city_day.csv or generates synthetic data for demonstration
- Preprocesses the data and handles missing values
- Calculates AQI values from pollutant concentrations
- Trains both Random Forest and XGBoost models
- Evaluates models and selects the best performer
- Saves the trained model and feature information
- Generates performance visualizations

### 3. Launch the Web Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The application will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://172.16.0.2:8501

The application will automatically open in your default web browser.

## 📊 Dataset Information

### Expected Dataset Format

The system expects a CSV file named city_day.csv in the data/ directory with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| PM2.5  | Particulate Matter 2.5 | µg/m³ |
| PM10   | Particulate Matter 10 | µg/m³ |
| NO     | Nitric Oxide | µg/m³ |
| NO2    | Nitrogen Dioxide | µg/m³ |
| NH3    | Ammonia | µg/m³ |
| CO     | Carbon Monoxide | mg/m³ |
| SO2    | Sulfur Dioxide | µg/m³ |
| O3     | Ozone | µg/m³ |

### Data Sources

You can obtain air quality datasets from:
- [Kaggle Air Quality Datasets](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- Government environmental agencies
- OpenAQ platform
- Local air quality monitoring stations

**Note**: If no dataset is provided, the system will automatically generate synthetic data for demonstration purposes.

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Handling**: Uses forward-fill and backward-fill methods
- **AQI Calculation**: Implements a simplified AQI calculation based on pollutant normalization
- **Feature Selection**: Automatically selects available pollutant columns

### 2. Model Training
- **Algorithms**: Random Forest Regressor and XGBoost Regressor
- **Evaluation Metrics**: Root Mean Squared Error (RMSE) and R² Score
- **Model Selection**: Automatically selects the model with the highest R² score

### 3. Model Evaluation
- Train/test split (80/20)
- Performance comparison between models
- Visualization of results and feature correlations

## 🖥️ Web Application Features

### Interactive Dashboard
- **Sidebar Controls**: Input fields for all pollutant concentrations
- **Real-time Predictions**: Instant AQI calculation and categorization
- **Health Impact Assessment**: Color-coded warnings based on AQI levels
- **Gauge Visualization**: Interactive AQI gauge with category zones

### Visualizations
- **Historical Trends**: Sample AQI trend over time
- **AQI Categories**: Reference table with health implications
- **Input Summary**: Review of entered pollutant values

### AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | Good 🟢 | Satisfactory air quality |
| 51-100 | Moderate 🟡 | Acceptable for most people |
| 101-150 | Unhealthy for Sensitive Groups 🟠 | Sensitive individuals may experience effects |
| 151-200 | Unhealthy 🔴 | General public may experience effects |
| 201-300 | Very Unhealthy 🟣 | Health alert for everyone |
| 301-500 | Hazardous 🔴 | Emergency conditions |

## 🔧 Configuration and Customization

### Model Parameters
You can modify the following parameters in model_training.py:
- 
_estimators: Number of trees in the forest/boosting rounds
- 	est_size: Train/test split ratio
- andom_state: Random seed for reproducibility

### Web Application Settings
Customize the Streamlit app in pp.py:
- Default input values
- Color schemes
- Chart configurations
- Page layout options

## 📈 Model Performance

The system automatically evaluates models using:
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **R² Score**: Explains variance in the data
- **Cross-validation**: Ensures model robustness

Performance metrics are displayed during training and saved as visualizations.

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages using pip install -r requirements.txt

2. **Model file not found**: Run python model_training.py before launching the web app

3. **Data file not found**: Place your dataset in the data/ folder or let the system use synthetic data

4. **Port already in use**: Specify a different port: streamlit run app.py --server.port 8502

### System Requirements
- Python 3.7 or higher
- At least 4GB RAM for model training
- Modern web browser for the Streamlit interface

## 📝 Usage Examples

### Command Line Training
`ash
# Basic training
python model_training.py

# View training progress
python model_training.py > training_log.txt
`

### Web Application Usage
1. Open the Streamlit app
2. Enter pollutant concentrations in the sidebar
3. Click "Predict AQI"
4. View results and health recommendations

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new machine learning algorithms
- Improving the web interface
- Enhancing data preprocessing methods
- Adding more visualization options
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

## 🔗 References

- [EPA Air Quality Index Guide](https://www.airnow.gov/aqi/aqi-basics/)
- [WHO Air Quality Guidelines](https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Ensure all dependencies are properly installed
4. Verify your data format matches the expected structure

---

**Note**: This system is designed for educational and demonstration purposes. For production use in critical applications, additional validation and calibration with official air quality standards would be recommended.
