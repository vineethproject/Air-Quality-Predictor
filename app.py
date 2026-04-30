import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_features():
    """
    Load the trained model and feature columns.
    
    Returns:
    tuple: (model, feature_columns)
    """
    try:
        model = joblib.load('models/aqi_predictor_model.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please run model_training.py first to train the model.")
        st.stop()

def get_aqi_category_and_color(aqi_value):
    """
    Get AQI category and corresponding color based on AQI value.
    
    Parameters:
    aqi_value (float): Predicted AQI value
    
    Returns:
    tuple: (category, color, health_message)
    """
    if aqi_value <= 50:
        return "Good", "#00E400", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "#FFFF00", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi_value <= 200:
        return "Unhealthy", "#FF0000", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8F3F97", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "#7E0023", "Health warning of emergency conditions: everyone is more likely to be affected."

def create_aqi_gauge(aqi_value):
    """
    Create a gauge chart for AQI visualization.
    
    Parameters:
    aqi_value (float): AQI value to display
    
    Returns:
    plotly.graph_objects.Figure: Gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Air Quality Index (AQI)"},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, 500]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "#00E400"},
                {'range': [50, 100], 'color': "#FFFF00"},
                {'range': [100, 150], 'color': "#FF7E00"},
                {'range': [150, 200], 'color': "#FF0000"},
                {'range': [200, 300], 'color': "#8F3F97"},
                {'range': [300, 500], 'color': "#7E0023"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 200
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def create_sample_historical_data():
    """
    Create sample historical AQI data for visualization.
    
    Returns:
    pd.DataFrame: Sample historical data
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Create some realistic AQI variation
    base_aqi = 75
    aqi_values = []
    
    for i, date in enumerate(dates):
        # Add some seasonal and random variation
        seasonal_factor = 10 * np.sin(i * 2 * np.pi / 30)  # 30-day cycle
        random_factor = np.random.normal(0, 15)
        daily_aqi = max(0, base_aqi + seasonal_factor + random_factor)
        aqi_values.append(daily_aqi)
    
    return pd.DataFrame({
        'Date': dates,
        'AQI': aqi_values
    })

def main():
    """
    Main Streamlit application.
    """
    # Title and header
    st.title("🌍 Air Quality Index (AQI) Prediction System")
    st.markdown("---")
    
    # Load model and features
    try:
        model, feature_columns = load_model_and_features()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run python model_training.py first to train and save the model.")
        st.stop()
    
    # Sidebar for input parameters
    st.sidebar.header("🔧 Enter Pollutant and Weather Data")
    st.sidebar.markdown("Adjust the values below to predict the Air Quality Index:")
    
    # Create input fields for each feature
    user_inputs = {}
    
    # Default values for demonstration
    default_values = {
        'PM2.5': 35.0,
        'PM10': 60.0,
        'NO': 25.0,
        'NO2': 40.0,
        'NH3': 20.0,
        'CO': 1.2,
        'SO2': 15.0,
        'O3': 55.0
    }
    
    # Units for each pollutant
    units = {
        'PM2.5': 'µg/m³',
        'PM10': 'µg/m³',
        'NO': 'µg/m³',
        'NO2': 'µg/m³',
        'NH3': 'µg/m³',
        'CO': 'mg/m³',
        'SO2': 'µg/m³',
        'O3': 'µg/m³'
    }
    
    for feature in feature_columns:
        default_val = default_values.get(feature, 50.0)
        unit = units.get(feature, 'units')
        
        user_inputs[feature] = st.sidebar.number_input(
            f"{feature} ({unit})",
            min_value=0.0,
            max_value=1000.0,
            value=default_val,
            step=0.1,
            help=f"Enter the {feature} concentration"
        )
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Predict AQI", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            # Create input dataframe
            input_data = pd.DataFrame([user_inputs])
            
            # Make prediction
            try:
                predicted_aqi = model.predict(input_data)[0]
                
                # Display prediction
                st.subheader("🎯 Prediction Results")
                
                # Get category and color
                category, color, health_message = get_aqi_category_and_color(predicted_aqi)
                
                # Display AQI value with color coding
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; margin: 20px 0;'>
                    <h1 style='color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin: 0;'>
                        AQI: {predicted_aqi:.1f}
                    </h1>
                    <h3 style='color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin: 10px 0 0 0;'>
                        {category}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Health message
                st.info(f"**Health Impact:** {health_message}")
                
                # Show alert for unhealthy conditions
                if predicted_aqi > 100:
                    st.warning("⚠️ **Alert:** Air quality is unhealthy! Consider limiting outdoor activities.")
                elif predicted_aqi > 200:
                    st.error("🚨 **Danger:** Air quality is hazardous! Avoid outdoor activities and consider staying indoors.")
                
                # Display gauge chart
                gauge_fig = create_aqi_gauge(predicted_aqi)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Show input summary
                with st.expander("📊 Input Data Summary"):
                    input_df = pd.DataFrame(list(user_inputs.items()), columns=['Pollutant', 'Value'])
                    input_df['Unit'] = input_df['Pollutant'].map(units)
                    st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        else:
            st.info("👈 Enter pollutant values in the sidebar and click 'Predict AQI' to get started!")
            
            # Show sample data input
            st.subheader("📝 Sample Input Values")
            st.write("Here are some typical pollutant concentration ranges:")
            
            sample_data = []
            for feature in feature_columns:
                unit = units.get(feature, 'units')
                if feature == 'PM2.5':
                    range_info = "0-35 (Good), 35-75 (Moderate), >75 (Unhealthy)"
                elif feature == 'PM10':
                    range_info = "0-50 (Good), 50-150 (Moderate), >150 (Unhealthy)"
                elif feature == 'CO':
                    range_info = "0-2 (Good), 2-5 (Moderate), >5 (Unhealthy)"
                else:
                    range_info = "Varies by location and conditions"
                
                sample_data.append({
                    'Pollutant': feature,
                    'Unit': unit,
                    'Typical Range': range_info
                })
            
            sample_df = pd.DataFrame(sample_data)
            st.dataframe(sample_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Historical AQI Trend")
        
        # Create and display sample historical data
        historical_data = create_sample_historical_data()
        
        fig = px.line(historical_data, x='Date', y='AQI', 
                      title='Past 30 Days AQI Trend (Sample Data)',
                      color_discrete_sequence=['#1f77b4'])
        
        # Add AQI category zones
        fig.add_hline(y=50, line_dash="dash", line_color="green", 
                      annotation_text="Good", annotation_position="right")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", 
                      annotation_text="Moderate", annotation_position="right")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", 
                      annotation_text="Unhealthy for Sensitive", annotation_position="right")
        fig.add_hline(y=200, line_dash="dash", line_color="red", 
                      annotation_text="Unhealthy", annotation_position="right")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # AQI Categories Info
        st.subheader("🏷️ AQI Categories")
        categories_data = [
            {"AQI Range": "0-50", "Category": "Good", "Color": "🟢"},
            {"AQI Range": "51-100", "Category": "Moderate", "Color": "🟡"},
            {"AQI Range": "101-150", "Category": "Unhealthy for Sensitive Groups", "Color": "🟠"},
            {"AQI Range": "151-200", "Category": "Unhealthy", "Color": "🔴"},
            {"AQI Range": "201-300", "Category": "Very Unhealthy", "Color": "🟣"},
            {"AQI Range": "301-500", "Category": "Hazardous", "Color": "🔴"}
        ]
        
        categories_df = pd.DataFrame(categories_data)
        st.dataframe(categories_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌱 <strong>Air Quality Prediction System</strong> | Built with Streamlit & Machine Learning</p>
        <p><em>Note: This is a demonstration system. For official AQI data, please consult your local environmental agency.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
