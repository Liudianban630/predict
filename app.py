from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
import shap
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import pandas as pd

LRmodel = joblib.load('LRmodel.pkl')
X_test = pd.read_pickle('X_test.pkl')

# Load SHAP explainer once to avoid recomputation
explainer = shap.Explainer(LRmodel.predict_proba, X_test)

# App title and description
st.title("Prediction of the prognosis of patients after acute cerebral infarction")

# Input features
NIHSS_24h   = st.number_input('NIHSS_24h', min_value=0.0, max_value=42.0)
NIHSS_change= st.number_input('NIHSS_change', min_value=-42.0, max_value=42.0)
D_dimer     = st.number_input('D-dimer', min_value=0.0, max_value=20.0)
N           = st.number_input('N', min_value=0.0, max_value=100.0)
L_pct_24h   = st.number_input('L_pct_24h', min_value=0.0, max_value=100.0)
LOS         = st.number_input('LOS', min_value=0.0, max_value=365.0)

# Combine the input features into a NumPy array
input_features = np.array([[NIHSS_24h, NIHSS_change, D_dimer, N, L_pct_24h, LOS]])

# List the feature names corresponding to the input features
feature_names = ['NIHSS_24h', 'NIHSS_change', 'D-dimer', 'N', 'L_pct_24h', 'LOS']

# Convert the input features into a pandas DataFrame with proper column names
input_df = pd.DataFrame(input_features, columns=feature_names)

# Prediction and SHAP visualization
if st.button('predict'):
    # Prediction and probability
    prediction = LRmodel.predict(input_df)
    prediction_proba = LRmodel.predict_proba(input_df)[0][1]  # Probability for positive class

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