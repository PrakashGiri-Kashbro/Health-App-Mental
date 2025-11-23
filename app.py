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
            "Area Chart",
            "Histogram",
            "Scatter Plot"
        ]
    )
    
    # --- Visualization Sub-sections ---

    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")

        numeric_data = data.select_dtypes(include=['number'])
        corr = numeric_data.drop("stress_code", axis=1).corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
        st.pyplot(fig)

    elif viz_type == "Line Chart":
        st.line_chart(data.drop(["stress_level", "stress_code"], axis=1))

    elif viz_type == "Bar Chart":
        feature = st.selectbox("Select feature:", data.columns[:-2])
        st.bar_chart(data[feature])

    elif viz_type == "Area Chart":
        st.area_chart(data.drop(["stress_level", "stress_code"], axis=1))

    elif viz_type == "Histogram":
        feature = st.selectbox("Select numeric feature:", data.columns[:-2])
        fig, ax = plt.subplots()
        ax.hist(data[feature], bins=20, edgecolor='black')
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    elif viz_type == "Scatter Plot":
        x_axis = st.selectbox("X-axis:", data.columns[:-2], index=3) # Default to work_stress
        y_axis = st.selectbox("Y-axis:", data.columns[:-2], index=5) # Default to mood_score
        
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data['stress_level'], palette={'low': 'green', 'medium': 'orange', 'high': 'red'}, ax=ax)
        ax.set_title(f'Scatter Plot of {x_axis.title()} vs {y_axis.title()}')
        st.pyplot(fig)

elif menu == "Train Model Summary":
    st.header("🧠 Model Training Summary: Feature Importance")

    X = data.drop(["stress_level", "stress_code"], axis=1)
    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Relative Importance of Features in Predicting Stress Level")
    
    # Custom Bar Chart for Feature Importance
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance from Random Forest Model")
    st.pyplot(fig)
    
    st.dataframe(importance_df, use_container_width=True)
    st.markdown("> **Interpretation:** Features with higher importance contribute more significantly to the model's prediction of the stress level.")

# --- Enhanced Prediction Section (Bhutan Theme) ---
elif menu == "Predict Stress Level":
    st.header("🐲 Predict Stress Level")
    st.markdown("Please input your current well-being factors to receive a stress level prediction. *Yul-khor-tshul* (Take care of yourself).")

    # Input sliders 
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age (Years)', 15, 60, 25)
        sleep_hours = st.slider('Sleep Hours', 4.0, 10.0, 7.0, step=0.5)
        social_interaction = st.slider('Social Interaction (0:Low - 7:High)', 0, 7, 3)

    with col2:
        work_stress = st.slider('Work/Study Stress (1:Low - 10:High)', 1, 10, 5)
        physical_activity = st.slider('Physical Activity (0:Low - 6:High)', 0, 6, 2)
        mood_score = st.slider('Mood Score (1:Poor - 10:Excellent)', 1, 10, 6)

    # Prepare features for prediction
    features_input = np.array([[age, sleep_hours, social_interaction, work_stress, physical_activity, mood_score]])

    if st.button("Predict Stress Level", use_container_width=True):
        prediction = model.predict(features_input)[0]
        
        # Define colors and emojis based on prediction
        if prediction == 'low':
            color = 'green'
            icon = '😌'
            description = 'Your predicted stress level is **LOW**. Continue to nurture your positive well-being habits!'
            bhutan_color = '#006B3E' # Green for low stress (GNH theme)
        elif prediction == 'medium':
            color = 'orange'
            icon = '🤔'
            description = 'Your predicted stress level is **MEDIUM**. It might be a good time to reflect on your balance and make minor adjustments.'
            bhutan_color = '#E9900B' # Orange/Red for medium stress (Flag theme)
        else: # high
            color = 'red'
            icon = '😟'
            description = 'Your predicted stress level is **HIGH**. Please consider prioritizing rest, managing stressors, and seeking support.'
            bhutan_color = '#BC3A3A' # Red for high stress

        st.markdown("---")
        
        # Display Prediction Result in a stylized box
        st.markdown(
            f"""
            <div style="background-color: {bhutan_color}; padding: 15px; border-radius: 10px; color: white;">
                <h3 style="color: white; margin-top: 0px;">{icon} Prediction Result:</h3>
                <h1 style="color: white; text-align: center;">{prediction.upper()}</h1>
                <p style="color: white; font-size: 1.1em;">{description}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Create the "Bhutan Flag" styled table with input features 
        st.subheader("Input Features Summary 🇧🇹")
        
        # Define HTML/CSS for a two-color table (Saffron and Orange/Red stripes)
        bhutan_style_table = f"""
        <style>
        .bhutan-table {{
            border-collapse: collapse; 
            width: 100%; 
            margin-top: 15px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }}
        .bhutan-table th, .bhutan-table td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        .bhutan-table-header th {{
            background-color: #FFC400; /* Saffron/Yellow */
            color: black;
            font-size: 1.1em;
        }}
        .bhutan-table-row:nth-child(even) {{
            background-color: #FF8C00; /* Orange/Red */
            color: black;
        }}
        .bhutan-table-row:nth-child(odd) {{
            background-color: #FEE7AC; /* Light Yellow */
            color: black;
        }}
        </style>
        
        <table class="bhutan-table">
            <thead>
                <tr class="bhutan-table-header">
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr class="bhutan-table-row">
                    <td>Age (Years)</td>
                    <td>{age}</td>
                </tr>
                <tr class="bhutan-table-row">
                    <td>Sleep Hours</td>
                    <td>{sleep_hours}</td>
                </tr>
                <tr class="bhutan-table-row">
                    <td>Social Interaction (0-7)</td>
                    <td>{social_interaction}</td>
                </tr>
                <tr class="bhutan-table-row">
                    <td>Work/Study Stress (1-10)</td>
                    <td>{work_stress}</td>
                </tr>
                <tr class="bhutan-table-row">
                    <td>Physical Activity (0-6)</td>
                    <td>{physical_activity}</td>
                </tr>
                <tr class="bhutan-table-row">
                    <td>Mood Score (1-10)</td>
                    <td>{mood_score}</td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(bhutan_style_table, unsafe_allow_html=True)
