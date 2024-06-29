import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Load your dataset and model here
df = pd.read_csv("Final_File.csv")
X = df[['B10_AVG_BATTERY_TEMP_PS_10B', 'B10_QUENCH_DUR_SEC_x', 'B10_TIME_TO_REACH_PEAK_POINT_x', 'B11_AVG_CHG_FORCE_y', 'B11_DURATION_EMPTY_OVENS_y', 'Blend_MAX_EXPANSION']]
y = df['AMS_IBF']

rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X, y)

# Function to categorize AMS value
def get_ams_status(ams_value):
    if ams_value <= 42.5:
        return "Bad AMS"
    elif 42.5 < ams_value <= 43:
        return "Mid AMS"
    else:
        return "High AMS"

# Function to predict AMS value
def predict_ams(temp, quench_dur, time_to_peak, avg_chg_force, empty_ovens_dur, blend_expansion):
    input_data = np.array([[temp, quench_dur, time_to_peak, avg_chg_force, empty_ovens_dur, blend_expansion]])
    predicted_ams = rf_regressor.predict(input_data)[0]
    return predicted_ams

# Streamlit interface
st.title("AMS Prediction Dashboard")

# Create input fields for each feature
temp = st.number_input("B10_AVG_BATTERY_TEMP_PS_10B", value=0.0)
quench_dur = st.number_input("B10_QUENCH_DUR_SEC_x", value=0.0)
time_to_peak = st.number_input("B10_TIME_TO_REACH_PEAK_POINT_x", value=0.0)
avg_chg_force = st.number_input("B11_AVG_CHG_FORCE_y", value=0.0)
empty_ovens_dur = st.number_input("B11_DURATION_EMPTY_OVENS_y", value=0.0)
blend_expansion = st.number_input("Blend_MAX_EXPANSION", value=0.0)

# Button to predict AMS
if st.button("Predict AMS"):
    if any(v == 0 for v in [temp, quench_dur, time_to_peak, avg_chg_force, empty_ovens_dur, blend_expansion]):
        st.error("All feature values must be non-zero.")
    else:
        predicted_ams = predict_ams(temp, quench_dur, time_to_peak, avg_chg_force, empty_ovens_dur, blend_expansion)
        ams_status = get_ams_status(predicted_ams)
        st.success(f'Predicted AMS: {predicted_ams:.2f}\nAMS Status: {ams_status}')
