import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained pipeline
try:
    full_pipeline = joblib.load("full_pipeline.pkl")
except FileNotFoundError:
    st.error("Error: full_pipeline.pkl not found. Please ensure the pipeline is saved correctly during training.")
    st.stop()

# Assuming the original feature columns used to train the pipeline are available
# This list should match the columns in X before preprocessing (after dropping PII and adding 'Last VL Result Clean')
# You need to ensure this list is correct based on your training data preparation.
# Let's infer them again as done previously, assuming the same initial cleaning steps.

original_cols = pd.read_csv("/content/Active on ART Patients Linelist_Aug-2025.csv", low_memory=False).columns.tolist()
pii_cols = [
    "MFL Code", "id", "Name", "CCC No", "NUPI",
    "SHA No", "Case Manager", "Establishment"
]
# Assuming '_target' was dropped from features, and 'Last VL Result' (raw) too
dropped_from_features = pii_cols + ["_target", "Last VL Result"] # Add any other columns dropped from X before preprocessing

# Infer features used for training X
feature_cols = [col for col in original_cols if col not in dropped_from_features]

# Re-derive numerical and categorical features for creating input widgets
# This is still a heuristic; saving and loading the actual lists is best.
# We use a temporary df to get the dtypes without loading the full data into memory unnecessarily
temp_df_for_inference = pd.read_csv("/content/Active on ART Patients Linelist_Aug-2025.csv", low_memory=False)[feature_cols]

numerical_features = temp_df_for_inference.select_dtypes(include=np.number).columns.tolist()
categorical_features = temp_df_for_inference.select_dtypes(include=['object', 'category']).columns.tolist()


# Streamlit App Interface
st.title("HIV Viral Load Suppression Prediction")
st.write("Enter patient details to predict viral load suppression status (1: Suppressed, 0: Not Suppressed)")

# Input Widgets (Create input fields for each feature)
input_data = {}

st.subheader("Numerical Features")
for feature in numerical_features:
    # Add specific widgets for key numerical features as needed
    if feature == 'Age at reporting':
        input_data[feature] = st.number_input(f"{feature}", min_value=0, max_value=150, value=30)
    elif feature in ['Weight', 'Height']:
         input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=60.0)
    elif feature == 'Last WHO Stage':
        input_data[feature] = st.number_input(f"{feature}", min_value=1, max_value=4, value=1)
    elif feature == 'Months of Prescription':
        input_data[feature] = st.number_input(f"{feature}", min_value=0, value=3)
    elif feature == 'Last risk score':
        input_data[feature] = st.number_input(f"{feature}", min_value=0, value=0)
    elif feature == 'Last VL Result Clean':
        # This feature was engineered, handle input carefully or don't include direct input if it's derived.
        # If you do include it, ensure it matches the expected range/format.
        # For now, let's add a generic number input and explicitly convert to float.
         input_data[feature] = float(st.number_input(f"{feature}", min_value=0.0, value=500.0))
    else: # Handle other numerical features generically
        input_data[feature] = st.number_input(f"{feature}", value=0.0)


st.subheader("Categorical Features")
# Categorical Inputs (This is complex as you need options for each)
# Ideally, load unique values from training data to populate selectbox options.
# For demonstration, let's add a few example categorical inputs with placeholder options.
# You need to replicate the exact options and column names from your training data.

# Example for 'Sex' (assuming Sex was in categorical_features and has options 'M', 'F', 'missing')
if 'Sex' in categorical_features:
    input_data['Sex'] = st.selectbox("Sex", options=['M', 'F', 'missing'], index=2 if 'missing' in ['M', 'F', 'missing'] else 0) # Default to 'missing' if an option

# Example for 'Current Regimen'
if 'Current Regimen' in categorical_features:
     # This list should come from the unique values in your training data
    current_regimen_options = ['TDF/3TC/DTG', 'TDF/3TC/EFV', 'ABC/3TC/DTG', 'AZT/3TC/NVP', 'missing'] # Placeholder options - REPLACE with actual unique values from training
    input_data['Current Regimen'] = st.selectbox("Current Regimen", options=current_regimen_options, index=len(current_regimen_options)-1 if 'missing' in current_regimen_options else 0) # Default to 'missing'


# Create a dictionary for the remaining categorical features with a default 'missing' value
# This is a simplification; ideally, you'd have specific widgets for key features with relevant options.
for feature in categorical_features:
    if feature not in input_data: # Add only if not already added via a specific widget
         # For features without specific widgets, use a text input or selectbox with limited options
         # or assume a default. Using 'missing' as a default constant strategy.
         input_data[feature] = 'missing' # This assumes constant imputation strategy with 'missing'

# Ensure all features expected by the preprocessors are in the input_data dictionary
# This is important if you didn't create widgets for all features.
# Fill in any missing features with the default value used by the categorical imputer ('missing')
# or the mean for numerical (though numerical features should have widgets).
for feature in numerical_features:
    if feature not in input_data:
        # For numerical features without specific widgets, use a default value.
        # Using 0.0 as a generic placeholder; ideally use the mean or a more appropriate default.
        input_data[feature] = 0.0

for feature in categorical_features:
    if feature not in input_data:
        input_data[feature] = 'missing'


# Prediction Button
if st.button("Predict"):
    # Prepare input data for prediction
    # Create a DataFrame with all expected feature columns in the correct order
    input_df = pd.DataFrame([input_data], columns=feature_cols)

    # Use the loaded full_pipeline to make predictions
    # The pipeline handles all preprocessing steps internally
    try:
        prediction = full_pipeline.predict(input_df)
        prediction_proba = full_pipeline.predict_proba(input_df)[:, 1]

        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"Predicted Status: Suppressed (Probability: {prediction_proba[0]:.4f})")
        else:
            st.error(f"Predicted Status: Not Suppressed (Probability: {prediction_proba[0]:.4f})")

        st.write("Note: This prediction is based on the provided inputs and the trained model.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values and try again. Ensure the input features match the expected format.")
