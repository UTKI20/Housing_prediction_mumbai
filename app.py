import gradio as gr
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# --- 1. GLOBAL ASSET LOADIng---

# Define paths (MUST match the names you used when saving)
MODEL_PATH = 'final_best_model.joblib'
SCALER_PATH = 'robust_scaler.joblib'
ENCODERS_PATH = 'label_encoders.joblib'

# Global variables to hold the loaded assets
BEST_MODEL = None
SCALER = None
LABEL_ENCODERS = None


def load_assets():
    """Loads the trained model, scaler, and encoders."""
    global BEST_MODEL, SCALER, LABEL_ENCODERS

    try:
        BEST_MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        LABEL_ENCODERS = joblib.load(ENCODERS_PATH)
        print("✓ All model assets loaded successfully.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Asset file not found: {e}")
        print("Please ensure your .joblib files are in the same directory.")
        exit()


load_assets()

# --- 2. MODEL CONFIGURATION ---

# The list of ALL features, in the EXACT order the model was trained on.
# We are assuming 17 original features + 2 engineered features = 19 features total
# IMPORTANT: This list must be updated to match your final model's feature order!
BASE_NUMERICAL_FEATURES = [
    'log_Area', 'No. of Bedrooms', 'SportsFacility', 'ClubHouse',
    'VaastuCompliant', '24X7Security', 'Resale', 'JoggingTrack',
    'LiftAvailable', 'BED', 'Refrigerator', 'DiningTable', 'IndoorGames',
    'AC', 'RainWaterHarvesting', 'Intercom',



]
ENGINEERED_FEATURES = ['log_price_per_sqft', 'log_area_per_bedroom']
CATEGORICAL_FEATURES = ['Location']  # This is the only Label-Encoded feature

FINAL_NUMERICAL_COLS_FOR_SCALER = BASE_NUMERICAL_FEATURES + ENGINEERED_FEATURES
FINAL_FEATURE_NAMES = FINAL_NUMERICAL_COLS_FOR_SCALER + [f'{CATEGORICAL_FEATURES[0]}_encoded']

# Define ALL features and their default values if they aren't exposed in the UI
# This is crucial for matching the model's feature count.
RAW_FEATURE_DEFAULTS = {
    'Area': 1000,  # Original Area needed for engineering
    'No. of Bedrooms': 2,
    'SportsFacility': 0,
    'ClubHouse': 0,
    'VaastuCompliant': 0,
    '24X7Security': 1,
    'Resale': 0,
    'JoggingTrack': 0,
    'LiftAvailable': 1,
    'BED': 0,
    'Refrigerator': 0,
    'DiningTable': 0,
    'IndoorGames': 0,
    'AC': 0,
    'RainWaterHarvesting': 0,
    'Intercom': 1,
}


# --- 3. PREDICTION FUNCTION ---

## --- 3. PREDICTION FUNCTION (DEBUG VERSION) ---

def predict_price(area_sqft, bedrooms, location_name, has_sports_facility, has_club_house, has_ac):
    """
    Takes user inputs, transforms them, and returns the predicted price.
    """

    # 1. Start with the default raw feature values
    data = RAW_FEATURE_DEFAULTS.copy()

    # 2. Update with User Input values and ensure data types are correct
    data['Area'] = float(area_sqft) # Ensure float type
    data['No. of Bedrooms'] = float(bedrooms) # Ensure float type

    # Convert Yes/No UI inputs to 1.0 or 0.0 (match RobustScaler expectation)
    data['SportsFacility'] = 1.0 if has_sports_facility == 'Yes' else 0.0
    data['ClubHouse'] = 1.0 if has_club_house == 'Yes' else 0.0
    data['AC'] = 1.0 if has_ac == 'Yes' else 0.0

    # Create a DataFrame for processing
    input_df = pd.DataFrame([data])

    # 3. Apply necessary transformations

    # a. Log Transformation
    input_df['log_Area'] = np.log1p(input_df['Area'])

    # b. Engineered Features (check for division by zero)
    try:
        input_df['log_price_per_sqft'] = np.log1p(
            input_df['Area'].iloc[0] / input_df['No. of Bedrooms'].iloc[0]
        )
        input_df['log_area_per_bedroom'] = np.log1p(
            input_df['Area'].iloc[0] / input_df['No. of Bedrooms'].iloc[0]
        )
    except ZeroDivisionError:
        return "Error: Number of bedrooms cannot be zero."

    # c. Drop the original Area (which is now log_Area)
    input_df = input_df.drop('Area', axis=1)

    # d. Label Encoding for Location
    le = LABEL_ENCODERS['Location']
    try:
        loc_encoded_value = le.transform([location_name])[0]
    except ValueError:
        # Assign a safe, seen value if the input location is new
        loc_encoded_value = le.transform([le.classes_[0]])[0]

    input_df['Location_encoded'] = float(loc_encoded_value) # Ensure float type

    # 4. Prepare Final Input Array

    # a. Extract Numerical features for scaling
    numerical_data = input_df[FINAL_NUMERICAL_COLS_FOR_SCALER]

    # b. Scale the numerical features
    scaled_numerical = SCALER.transform(numerical_data)

    # c. Extract the encoded categorical feature(s)
    encoded_categorical = input_df[['Location_encoded']].values

    # d. Stack to create the final array
    final_input_array = np.hstack((scaled_numerical, encoded_categorical))

    # --- DEBUGGING OUTPUT ---
    print(f"\nDebug: Model expects {len(FINAL_FEATURE_LIST)} features.")
    print(f"Debug: Input array shape (rows, cols): {final_input_array.shape}")

    if final_input_array.shape[1] != len(FINAL_FEATURE_LIST):
        return f"CRITICAL ERROR: Feature count mismatch. Model expects {len(FINAL_FEATURE_LIST)} features but received {final_input_array.shape[1]}."
    # --- END DEBUGGING ---

    # 5. Predict and Inverse Transform
    log_pred = BEST_MODEL.predict(final_input_array)[0]

    # Inverse transform
    price_in_inr = np.expm1(log_pred)
    price_in_crores = price_in_inr / 10000000

    # 6. Format Output
    return f"Predicted Price: ₹{price_in_crores:,.2f} Crores"


# --- 4. GRADIO INTERFACE DEFINITION ---

# Get location choices from the LabelEncoder classes
# Assume 'Location' is the key in the encoders dictionary
try:
    LOCATION_CHOICES = list(LABEL_ENCODERS['Location'].classes_)
except (KeyError, TypeError):
    LOCATION_CHOICES = ['Bandra', 'Khar', 'Andheri', 'Worli']  # Fallback if loading fails

# Define the components that match the arguments of your predict_price function
iface_inputs = [
    gr.Slider(minimum=200, maximum=8000, value=1000, step=50, label="Area (Sq. Ft.)"),
    gr.Slider(minimum=1, maximum=6, value=2, step=1, label="No. of Bedrooms"),
    gr.Dropdown(choices=LOCATION_CHOICES, value=LOCATION_CHOICES[0], label="Location"),
    gr.Radio(choices=['Yes', 'No'], value='Yes', label="Sports Facility"),
    gr.Radio(choices=['Yes', 'No'], value='Yes', label="Club House"),
    gr.Radio(choices=['Yes', 'No'], value='No', label="Air Conditioning (AC)"),
]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=iface_inputs,
    outputs="text",
    title="Mumbai Housing Price Predictor (Tuned Ensemble Model)",
 description="Estimate property value in Crores based on features and amenities in Mumbai."
)

# Launch the App
if __name__ == "__main__":
    iface.launch(share=False)