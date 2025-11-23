# --- Modified HTML Table Styling for Bolder Colors ---
elif menu == "Predict Stress Level":
    st.header("🐲 Predict Stress Level")
    st.markdown("Please input your current well-being factors to receive a stress level prediction. *Yul-khor-tshul* (Take care of yourself).")

    # Input sliders (kept the same for brevity)
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
        
        # Define colors and emojis based on prediction (Kept the same)
        if prediction == 'low':
            color = 'green'
            icon = '😌'
            description = 'Your predicted stress level is **LOW**. Continue to nurture your positive well-being habits!'
            bhutan_color = '#006B3E' 
        elif prediction == 'medium':
            color = 'orange'
            icon = '🤔'
            description = 'Your predicted stress level is **MEDIUM**. It might be a good time to reflect on your balance and make minor adjustments.'
            bhutan_color = '#E9900B'
        else: # high
            color = 'red'
            icon = '😟'
            description = 'Your predicted stress level is **HIGH**. Please consider prioritizing rest, managing stressors, and seeking support.'
            bhutan_color = '#BC3A3A'

        st.markdown("---")
        
        # Display Prediction Result in a stylized box (Kept the same)
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

        # --- Enhanced Table with Bolder Flag Colors ---
        st.subheader("Input Features Summary 🇧🇹")
        
        # Define HTML/CSS for a two-color table (Saffron and Orange/Red stripes) with BOLDER COLORS
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
            background-color: #FFB300; /* BOLDER Saffron/Yellow */
            color: black;
            font-size: 1.1em;
        }}
        .bhutan-table-row:nth-child(even) {{
            background-color: #FF6F00; /* BOLDER Orange/Red */
            color: white; /* Changed text color to white for contrast */
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
