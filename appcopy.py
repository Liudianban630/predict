import lightgbm as lgb
import shap
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved LGBM model
LGBM_model = joblib.load('lgbm_model.pkl')
X_test = pd.read_pickle('X_test.pkl')

# Load SHAP explainer once to avoid recomputation
explainer = shap.Explainer(LGBM_model.predict_proba, X_test)

# App title and description
st.title("Prediction of the prognosis of patients after acute cerebral infarction")

# Input features
LOS         = st.number_input('LOS', min_value=0.0, max_value=365.0)
Age         = st.number_input('Age', min_value=0.0, max_value=100.0)
Ddimer      = st.number_input('Ddimer', min_value=0.0, max_value=20.0)
M_pct_24h   = st.number_input('M_pct_24h', min_value=0.0, max_value=100.0)
L_24h       = st.number_input('L_24h', min_value=0.0, max_value=8.0)
RBC_24h     = st.number_input('RBC_24h', min_value=0.0, max_value=7.0)
HB_24h      = st.number_input('HB_24h', min_value=0.0, max_value=200.0)
NIHSS_24h   = st.number_input('NIHSS_24h', min_value=0.0, max_value=42.0)
NHISSchange = st.number_input('NHISSchange', min_value=-42.0, max_value=42.0, value=0.0)

# Combine the input features into a NumPy array
input_features = np.array([[LOS, Age, Ddimer, M_pct_24h,L_24h, RBC_24h, HB_24h, NIHSS_24h, NHISSchange]])

# List the feature names corresponding to the input features
feature_names = ['LOS', 'Age', 'Ddimer', 'M_pct_24h', 'L_24h', 'RBC_24h', 'HB_24h', 'NIHSS_24h', 'NHISSchange']

# Convert the input features into a pandas DataFrame with proper column names
input_df = pd.DataFrame(input_features, columns=feature_names)
assert list(input_df.columns) == feature_names, "Feature order does not match!"
# Prediction and SHAP visualization
if st.button('predict'):
    # Prediction and probability
    prediction = LGBM_model.predict(input_df)
    prediction_proba = LGBM_model.predict_proba(input_df)[0][1]  # Probability for positive class

    # Display result
    html_string = f'<p style="font-size: 33px;">Probability of the unfavorable prognosis is {prediction_proba*100:.2f}%</p>'
    st.write(html_string, unsafe_allow_html=True)
        
    # SHAP visualization
    shap_values = explainer(input_df)
    shap_values = shap_values[..., 1]  # Select the SHAP values for the positive class

    # Initialize JavaScript environment for SHAP force plot
    shap.initjs()

    # Generate SHAP force plot
    force_plot = shap.plots.force(shap_values)

    # Custom method to display SHAP force plot in Streamlit
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st_shap(force_plot)