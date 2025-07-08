import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Solar Panel Performance Analysis",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">☀️ Solar Panel Performance Analysis System</h1>', unsafe_allow_html=True)
st.markdown("### Analyze solar panel performance across different seasons with machine learning")

# Feature ranges for data generation
feature_ranges = {
    'summer': {
        'irradiance': (600, 1000),
        'humidity': (10, 50),
        'wind_speed': (0, 5),
        'ambient_temperature': (30, 45),
        'tilt_angle': (10, 40),
    },
    'winter': {
        'irradiance': (300, 700),
        'humidity': (30, 70),
        'wind_speed': (1, 6),
        'ambient_temperature': (5, 20),
        'tilt_angle': (10, 40),
    },
    'monsoon': {
        'irradiance': (100, 600),
        'humidity': (70, 100),
        'wind_speed': (2, 8),
        'ambient_temperature': (20, 35),
        'tilt_angle': (10, 40),
    }
}

# Calculation functions
def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 
            0.1 * ambient_temp - 0.03 * abs(tilt_angle - 30))

def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 
            0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))

def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.15 * irradiance - 0.1 * humidity + 0.01 * wind_speed + 
            0.05 * ambient_temp - 0.04 * abs(tilt_angle - 30))

# Data generation function
@st.cache_data
def generate_seasonal_data():
    # Season months and days
    season_months = {
        'summer': {'March': 31, 'April': 30, 'May': 31, 'June': 30},
        'winter': {'November': 30, 'December': 31, 'January': 31, 'February': 28},
        'monsoon': {'July': 31, 'August': 31, 'September': 30, 'October': 31}
    }
    
    calc_functions = {
        'summer': calc_kwh_summer,
        'winter': calc_kwh_winter,
        'monsoon': calc_kwh_monsoon
    }
    
    all_data = []
    
    for season, months_days in season_months.items():
        for month, days in months_days.items():
            for day in range(1, days + 1):
                ranges = feature_ranges[season]
                
                irr = np.random.uniform(*ranges['irradiance'])
                hum = np.random.uniform(*ranges['humidity'])
                wind = np.random.uniform(*ranges['wind_speed'])
                temp = np.random.uniform(*ranges['ambient_temperature'])
                tilt = np.random.uniform(*ranges['tilt_angle'])
                
                kwh = calc_functions[season](irr, hum, wind, temp, tilt)
                
                all_data.append({
                    'irradiance': round(irr, 2),
                    'humidity': round(hum, 2),
                    'wind_speed': round(wind, 2),
                    'ambient_temperature': round(temp, 2),
                    'tilt_angle': round(tilt, 2),
                    'kwh': round(kwh, 2),
                    'season': season,
                    'month': month,
                    'day': day
                })
    
    return pd.DataFrame(all_data)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "📊 Data Generation", "📈 Data Visualization", "🤖 ML Models", "🔮 Predictions"]
)

# Generate or load data
if 'df' not in st.session_state:
    st.session_state.df = generate_seasonal_data()

df = st.session_state.df

# Home Page
if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to Solar Panel Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🌞 Features
        - **Seasonal Analysis**: Compare performance across Summer, Winter, and Monsoon
        - **Environmental Factors**: Analyze impact of irradiance, humidity, wind, and temperature
        - **Machine Learning**: Linear regression for energy prediction and classification for season detection
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Data Insights
        - **365 Days**: Complete yearly solar performance data
        - **5 Input Features**: Irradiance, humidity, wind speed, temperature, tilt angle
        - **3 Seasons**: Summer, Winter, and Monsoon patterns
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 Tools Used
        - **Streamlit**: Interactive web interface
        - **Scikit-learn**: Machine learning models
        - **Plotly**: Interactive visualizations
        - **Pandas**: Data manipulation
        """)
    
    # Key metrics
    st.markdown('<h3 class="sub-header">Dataset Overview</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        st.metric("Average kWh", f"{df['kwh'].mean():.2f}")
    
    with col3:
        st.metric("Max kWh", f"{df['kwh'].max():.2f}")
    
    with col4:
        st.metric("Seasons", df['season'].nunique())
    
    # Quick data preview
    st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

# Data Generation Page
elif page == "📊 Data Generation":
    st.markdown('<h2 class="sub-header">Data Generation & Statistics</h2>', unsafe_allow_html=True)
    
    # Regenerate data button
    if st.button("🔄 Regenerate Data", type="primary"):
        st.session_state.df = generate_seasonal_data()
        st.success("Data regenerated successfully!")
        st.rerun()
    
    # Data statistics
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Descriptive Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.write("**Data Types & Info:**")
        buffer = []
        buffer.append(f"Dataset Shape: {df.shape}")
        buffer.append(f"Memory Usage: {df.memory_usage().sum()} bytes")
        buffer.append("\nColumn Information:")
        for col in df.columns:
            buffer.append(f"- {col}: {df[col].dtype}")
        st.text("\n".join(buffer))
    
    # Season distribution
    st.markdown("### 🌍 Season Distribution")
    season_counts = df['season'].value_counts()
    
    fig = px.pie(values=season_counts.values, names=season_counts.index, 
                 title="Distribution of Records by Season")
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly distribution
    st.markdown("### 📅 Monthly Distribution")
    month_counts = df['month'].value_counts()
    
    fig = px.bar(x=month_counts.index, y=month_counts.values,
                 title="Number of Records by Month")
    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Records")
    st.plotly_chart(fig, use_container_width=True)

# Data Visualization Page
elif page == "📈 Data Visualization":
    st.markdown('<h2 class="sub-header">Data Visualization & Analysis</h2>', unsafe_allow_html=True)
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose visualization type:",
        ["kWh by Season", "Feature Correlations", "Seasonal Patterns", "Monthly Trends", "Feature Distributions"]
    )
    
    if viz_type == "kWh by Season":
        st.markdown("### ⚡ Energy Output by Season")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig = px.box(df, x='season', y='kwh', 
                        title="kWh Distribution by Season")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot
            fig = px.violin(df, x='season', y='kwh',
                           title="kWh Distribution (Violin Plot)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Season statistics
        season_stats = df.groupby('season')['kwh'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        st.write("**Season Statistics:**")
        st.dataframe(season_stats, use_container_width=True)
    
    elif viz_type == "Feature Correlations":
        st.markdown("### 🔗 Feature Correlations")
        
        # Correlation matrix
        numeric_cols = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix
        st.markdown("### 📊 Scatter Plot Matrix")
        selected_features = st.multiselect(
            "Select features for scatter plot:",
            numeric_cols,
            default=['irradiance', 'humidity', 'kwh']
        )
        
        if len(selected_features) >= 2:
            fig = px.scatter_matrix(df, dimensions=selected_features,
                                   color='season',
                                   title="Scatter Plot Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Seasonal Patterns":
        st.markdown("### 🌦️ Seasonal Patterns")
        
        # Feature comparison across seasons
        feature = st.selectbox(
            "Select feature to analyze:",
            ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']
        )
        
        fig = px.box(df, x='season', y=feature,
                    title=f"{feature.replace('_', ' ').title()} across Seasons")
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal averages
        seasonal_avg = df.groupby('season')[numeric_cols].mean().round(2)
        st.write("**Seasonal Averages:**")
        st.dataframe(seasonal_avg, use_container_width=True)
    
    elif viz_type == "Monthly Trends":
        st.markdown("### 📅 Monthly Trends")
        
        # Monthly kWh trends
        monthly_kwh = df.groupby('month')['kwh'].mean().reset_index()
        
        # Define month order
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_kwh['month'] = pd.Categorical(monthly_kwh['month'], categories=month_order, ordered=True)
        monthly_kwh = monthly_kwh.sort_values('month')
        
        fig = px.line(monthly_kwh, x='month', y='kwh',
                     title="Average kWh by Month",
                     markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Average kWh")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Feature Distributions":
        st.markdown("### 📊 Feature Distributions")
        
        # Select feature
        feature = st.selectbox(
            "Select feature for distribution:",
            numeric_cols
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=feature, nbins=30,
                              title=f"Distribution of {feature.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histogram by season
            fig = px.histogram(df, x=feature, color='season',
                              title=f"{feature.replace('_', ' ').title()} by Season")
            st.plotly_chart(fig, use_container_width=True)

# ML Models Page
elif page == "🤖 ML Models":
    st.markdown('<h2 class="sub-header">Machine Learning Models</h2>', unsafe_allow_html=True)
    
    model_type = st.selectbox(
        "Choose model type:",
        ["Linear Regression (kWh Prediction)", "Logistic Regression (Season Classification)"]
    )
    
    if model_type == "Linear Regression (kWh Prediction)":
        st.markdown("### 📈 Linear Regression for kWh Prediction")
        
        # Features and target
        X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
        y = df['kwh']
        
        # Train-test split
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if st.button("🚀 Train Linear Regression Model", type="primary"):
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Squared Error", f"{mse:.4f}")
            
            with col2:
                st.metric("R² Score", f"{r2:.4f}")
            
            with col3:
                st.metric("Training Size", len(X_train))
            
            # Feature importance
            st.markdown("### 🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            fig = px.bar(importance_df, x='Feature', y='Coefficient',
                        title="Feature Coefficients")
            st.plotly_chart(fig, use_container_width=True)
            
            # Actual vs Predicted
            st.markdown("### 🎯 Actual vs Predicted")
            
            fig = px.scatter(x=y_test, y=y_pred,
                           title="Actual vs Predicted kWh",
                           labels={'x': 'Actual kWh', 'y': 'Predicted kWh'})
            
            # Add diagonal line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                         line=dict(color="red", dash="dash"))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store model in session state
            st.session_state.regression_model = model
            st.success("Model trained successfully!")
    
    elif model_type == "Logistic Regression (Season Classification)":
        st.markdown("### 🌍 Logistic Regression for Season Classification")
        
        # Features and target
        X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']]
        y = df['season']
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Train-test split
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="log_test_size")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
        
        # Train model
        if st.button("🚀 Train Logistic Regression Model", type="primary"):
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            
            with col2:
                st.metric("Training Size", len(X_train))
            
            with col3:
                st.metric("Test Size", len(X_test))
            
            # Classification report
            st.markdown("### 📊 Classification Report")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
            
            # Confusion matrix
            st.markdown("### 🔍 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, 
                           x=le.classes_, 
                           y=le.classes_,
                           title="Confusion Matrix",
                           labels={'x': 'Predicted', 'y': 'Actual'},
                           text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Store model in session state
            st.session_state.classification_model = model
            st.session_state.label_encoder = le
            st.success("Model trained successfully!")

# Predictions Page
elif page == "🔮 Predictions":
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    prediction_type = st.selectbox(
        "Choose prediction type:",
        ["kWh Prediction", "Season Classification", "Batch Predictions"]
    )
    
    if prediction_type == "kWh Prediction":
        st.markdown("### ⚡ Predict Solar Panel Energy Output")
        
        if 'regression_model' in st.session_state:
            # Input features
            col1, col2 = st.columns(2)
            
            with col1:
                irradiance = st.slider("Irradiance (W/m²)", 100, 1000, 500)
                humidity = st.slider("Humidity (%)", 10, 100, 50)
                wind_speed = st.slider("Wind Speed (m/s)", 0.0, 8.0, 3.0, 0.1)
            
            with col2:
                ambient_temp = st.slider("Ambient Temperature (°C)", 5, 45, 25)
                tilt_angle = st.slider("Tilt Angle (degrees)", 10, 40, 30)
            
            # Make prediction
            if st.button("🔮 Predict kWh", type="primary"):
                input_data = np.array([[irradiance, humidity, wind_speed, ambient_temp, tilt_angle]])
                prediction = st.session_state.regression_model.predict(input_data)[0]
                
                st.success(f"Predicted Energy Output: **{prediction:.2f} kWh**")
                
                # Show input summary
                st.markdown("### 📋 Input Summary")
                input_df = pd.DataFrame({
                    'Feature': ['Irradiance', 'Humidity', 'Wind Speed', 'Ambient Temperature', 'Tilt Angle'],
                    'Value': [f"{irradiance} W/m²", f"{humidity}%", f"{wind_speed} m/s", f"{ambient_temp}°C", f"{tilt_angle}°"],
                    'Unit': ['W/m²', '%', 'm/s', '°C', 'degrees']
                })
                st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        else:
            st.warning("Please train the Linear Regression model first in the ML Models page.")
    
    elif prediction_type == "Season Classification":
        st.markdown("### 🌍 Predict Season from Solar Data")
        
        if 'classification_model' in st.session_state:
            # Input features
            col1, col2, col3 = st.columns(3)
            
            with col1:
                irradiance = st.slider("Irradiance (W/m²)", 100, 1000, 500, key="class_irr")
                humidity = st.slider("Humidity (%)", 10, 100, 50, key="class_hum")
            
            with col2:
                wind_speed = st.slider("Wind Speed (m/s)", 0.0, 8.0, 3.0, 0.1, key="class_wind")
                ambient_temp = st.slider("Ambient Temperature (°C)", 5, 45, 25, key="class_temp")
            
            with col3:
                tilt_angle = st.slider("Tilt Angle (degrees)", 10, 40, 30, key="class_tilt")
                kwh = st.slider("kWh Output", 0.0, 300.0, 150.0, 1.0)
            
            # Make prediction
            if st.button("🔮 Predict Season", type="primary"):
                input_data = np.array([[irradiance, humidity, wind_speed, ambient_temp, tilt_angle, kwh]])
                prediction = st.session_state.classification_model.predict(input_data)[0]
                prediction_proba = st.session_state.classification_model.predict_proba(input_data)[0]
                
                predicted_season = st.session_state.label_encoder.inverse_transform([prediction])[0]
                
                st.success(f"Predicted Season: **{predicted_season.title()}**")
                
                # Show probabilities
                st.markdown("### 📊 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Season': st.session_state.label_encoder.classes_,
                    'Probability': prediction_proba
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(prob_df, x='Season', y='Probability',
                           title="Season Prediction Probabilities")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please train the Logistic Regression model first in the ML Models page.")
    
    elif prediction_type == "Batch Predictions":
        st.markdown("### 📊 Batch Predictions")
        
        if 'regression_model' in st.session_state:
            # Upload CSV or use sample data
            uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    
                    # Check if required columns exist
                    required_cols = ['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']
                    
                    if all(col in batch_df.columns for col in required_cols):
                        # Make predictions
                        predictions = st.session_state.regression_model.predict(batch_df[required_cols])
                        batch_df['predicted_kwh'] = predictions
                        
                        st.success(f"Predictions completed for {len(batch_df)} records!")
                        
                        # Show results
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )
                    
                    else:
                        st.error(f"CSV must contain columns: {required_cols}")
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
            
            else:
                st.info("Upload a CSV file with the required columns to make batch predictions.")
                
                # Show sample format
                st.markdown("### 📋 Required CSV Format")
                sample_df = pd.DataFrame({
                    'irradiance': [600, 700, 800],
                    'humidity': [30, 40, 50],
                    'wind_speed': [2.0, 3.0, 4.0],
                    'ambient_temperature': [25, 30, 35],
                    'tilt_angle': [25, 30, 35]
                })
                st.dataframe(sample_df, use_container_width=True)
        
        else:
            st.warning("Please train the Linear Regression model first in the ML Models page.")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit • Solar Panel Performance Analysis System",
    unsafe_allow_html=True
)