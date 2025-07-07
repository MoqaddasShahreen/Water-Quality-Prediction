# Import all necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import altair as alt
import time

# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Interface configuration
st.set_page_config(page_title="Water Pollutants Predictor", layout="centered")

# Title and description
st.title("💧 Water Pollutants Predictor")
st.write(
    """
Predict pollutant levels in water samples based on Year and Station ID.
This tool helps environmental agencies monitor and improve water quality.
"""
)

# Sidebar input
st.sidebar.header("Input Parameters")

year_input = st.sidebar.number_input(
    "Enter Year",
    min_value=2000,
    max_value=2100,
    value=2022,
    step=1
)

station_id = st.sidebar.text_input("Enter Station ID", value="1")

# Predict button
if st.sidebar.button("Predict Pollution Levels"):
    if not station_id.strip():
        st.warning("⚠️ Please enter a valid Station ID.")
    else:
        # Progress bar animation
        progress = st.progress(0, text="Predicting pollutant levels...")
        for percent in range(0, 101, 10):
            progress.progress(percent, text=f"Loading... {percent}%")
            time.sleep(0.05)
        progress.empty()

        # Prepare input DataFrame
        input_df = pd.DataFrame({"year": [year_input], "id": [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=["id"])

        # Add any missing columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder columns to match model
        input_encoded = input_encoded[model_cols]

        # Make prediction
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ["O2", "NO3", "NO2", "SO4", "PO4", "CL"]

        # Create results DataFrame
        results_df = pd.DataFrame({
            "Pollutant": pollutants,
            "Predicted Level": [round(val, 2) for val in predicted_pollutants]
        })

        # Display results
        st.subheader(
            f"Predicted Pollutant Levels for Station '{station_id}' in {year_input}:"
        )
        st.table(results_df)

        # Horizontal bar chart visualization
        bar_chart = alt.Chart(results_df).mark_bar(color="#4dabf7").encode(
            x=alt.X("Predicted Level:Q", title="Predicted Level"),
            y=alt.Y("Pollutant:N", sort="-x", title="Pollutant"),
            tooltip=["Pollutant", "Predicted Level"]
        ).properties(
            width=600,
            height=300,
            title="Predicted Pollutant Levels"
        )

        st.altair_chart(bar_chart, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Developed by Moqaddas Shahreen.")
