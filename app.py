import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Bhutan Mental Health Predictor")

st.title("Mental Health Stress Level Predictor with Visualizations (Bhutan)")

@st.cache_resource
def load_data():
    """Generates synthetic dataset for demonstration."""
    np.random.seed(42)
    size = 600

    data = pd.DataFrame({
        'age': np.random.randint(15, 60, size),
        'sleep_hours': np.random.uniform(4, 10, size),
        'social_interaction': np.random.randint(0, 7, size),
        'work_stress': np.random.randint(1, 10, size),
        'physical_activity': np.random.randint(0, 6, size),
        'mood_score': np.random.randint(1, 10, size)
    })

    # Formula to derive a "Stress Score" for synthetic data generation
    # Stress Score increases with Work Stress and decreases with Mood/Physical Activity
    score = (data['work_stress'] * 0.5) + (10 - data['mood_score']) + (6 - data['physical_activity'])

    conditions = [
        (score < 8),
        ((score >= 8) & (score < 14)),
        (score >= 14)
    ]
    choices = ['low', 'medium', 'high']

    data['stress_level'] = np.select(conditions, choices, default='medium')

    return data


@st.cache_resource
def train_model(data):
    """Trains a RandomForestClassifier model."""
    # Convert target variable to category codes for model training
    data['stress_code'] = data['stress_level'].astype('category').cat.codes
    
    X = data.drop(["stress_level", "stress_code"], axis=1)
    y = data["stress_level"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model


data = load_data()
model = train_model(data)

# --- Navigation ---
menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations", "Train Model Summary", "Predict Stress Level"]
)

# --- Main Content ---

if menu == "Dataset Overview":
    st.header("📊 Dataset Overview")
    st.dataframe(data.head())
    
    st.header("📝 Summary of Data")
    st.dataframe(data.describe())
    
    st.header("🎯 Stress Level Distribution")
    # Use value_counts for the chart
    stress_counts = data["stress_level"].value_counts().sort_index(key=lambda x: x.map({'low': 0, 'medium': 1, 'high': 2}))
    st.bar_chart(stress_counts)

elif menu == "Visualizations":
    st.header("📈 Visualizations")

    viz_type = st.selectbox(
        "Choose chart type:",
        [
            "Correlation Heatmap",
            "Line Chart",
            "Bar Chart",
