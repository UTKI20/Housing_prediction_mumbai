

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("LOADING MUMBAI HOUSING PRICE PREDICTOR")
print("=" * 70)


# ================================
# 1. LOAD MODEL ARTIFACTS
# ================================

def load_model_assets():
    """Load all saved model components"""
    try:
        print("\nLoading model artifacts...")

        # Load model (best tuned model from training)
        model = joblib.load('best_model.joblib')
        print(f"  Model: {type(model).__name__}")

        # Load scaler (RobustScaler)
        scaler = joblib.load('scaler.joblib')
        print(f"  Scaler: {type(scaler).__name__}")

        # Load location encoder
        location_encoder = joblib.load('location_encoder.joblib')
        print(f"  Location Encoder: {len(location_encoder.classes_)} locations")

        # Load feature lists
        numerical_features = joblib.load('numerical_features.joblib')
        categorical_features = joblib.load('categorical_features.joblib')
        print(f"  Numerical Features: {len(numerical_features)}")
        print(f"  Categorical Features: {len(categorical_features)}")

        print("\nAll assets loaded successfully!")
        return model, scaler, location_encoder, numerical_features, categorical_features

    except FileNotFoundError as e:
        print(f"\nERROR: Missing file - {e}")
        print("\nRequired files:")
        print("  - best_model.joblib")
        print("  - scaler.joblib")
        print("  - location_encoder.joblib")
        print("  - numerical_features.joblib")
        print("  - categorical_features.joblib")
        print("\nPlease run the training script first to generate these files.")
        exit(1)


# Load assets
MODEL, SCALER, LOCATION_ENCODER, NUMERICAL_FEATURES, CATEGORICAL_FEATURES = load_model_assets()

# Get available locations
LOCATIONS = sorted(LOCATION_ENCODER.classes_.tolist())

# ================================
# 2. DEFAULT FEATURE VALUES
# ================================

FEATURE_DEFAULTS = {
    'No. of Bedrooms': 2.0,
    'Resale': 0.0,
    'MaintenanceStaff': 1.0,
    'Gymnasium': 1.0,
    'SwimmingPool': 1.0,
    'LandscapedGardens': 1.0,
    'JoggingTrack': 1.0,
    'RainWaterHarvesting': 1.0,
    'IndoorGames': 0.0,
    'ShoppingMall': 0.0,
    'Intercom': 1.0,
    'SportsFacility': 0.0,
    'ATM': 0.0,
    'ClubHouse': 0.0,
    'School': 0.0,
    '24X7Security': 1.0,
    'PowerBackup': 1.0,
    'CarParking': 1.0,
    'StaffQuarter': 0.0,
    'Cafeteria': 0.0,
    'MultipurposeRoom': 0.0,
    'Hospital': 0.0,
    'WashingMachine': 1.0,
    'Gasconnection': 1.0,
    'AC': 0.0,
    'Wifi': 1.0,
    "Children'splayarea": 1.0,
    'LiftAvailable': 1.0,
    'BED': 0.0,
    'VaastuCompliant': 0.0,
    'Microwave': 1.0,
    'GolfCourse': 0.0,
    'TV': 1.0,
    'DiningTable': 1.0,
    'Sofa': 1.0,
    'Wardrobe': 1.0,
    'Refrigerator': 1.0,
}


# ================================
# 3. PREDICTION FUNCTION
# ================================

def predict_house_price(area, bedrooms, location, sports, club, ac, gym, pool, security):
    """
    Predict house price based on user inputs
    Matches the exact preprocessing from training
    """

    try:
        # Validate inputs
        if area <= 0:
            return "Error: Area must be greater than 0"
        if bedrooms <= 0:
            return "Error: Number of bedrooms must be greater than 0"

        # Start with defaults
        features = FEATURE_DEFAULTS.copy()

        # Update with user inputs
        features['No. of Bedrooms'] = float(bedrooms)
        features['SportsFacility'] = 1.0 if sports == 'Yes' else 0.0
        features['ClubHouse'] = 1.0 if club == 'Yes' else 0.0
        features['AC'] = 1.0 if ac == 'Yes' else 0.0
        features['Gymnasium'] = 1.0 if gym == 'Yes' else 0.0
        features['SwimmingPool'] = 1.0 if pool == 'Yes' else 0.0
        features['24X7Security'] = 1.0 if security == 'Yes' else 0.0

        # Create DataFrame
        df = pd.DataFrame([features])

        # Feature Engineering (matching training)
        # 1. Price per sqft (use typical Mumbai rate for inference)
        df['price_per_sqft'] = 18000.0  # Typical rate
        df['log_price_per_sqft'] = np.log1p(df['price_per_sqft'])

        # 2. Area per bedroom
        df['area_per_bedroom'] = area / bedrooms
        df['log_area_per_bedroom'] = np.log1p(df['area_per_bedroom'])

        # 3. Log transform area
        df['log_Area'] = np.log1p(area)

        # Drop intermediate columns
        df = df.drop(['price_per_sqft', 'area_per_bedroom'], axis=1, errors='ignore')

        # Extract numerical features for scaling
        numerical_data = df[NUMERICAL_FEATURES].values

        # Scale numerical features
        scaled_numerical = SCALER.transform(numerical_data)

        # Encode location
        try:
            location_encoded = LOCATION_ENCODER.transform([location])[0]
        except ValueError:
            location_encoded = LOCATION_ENCODER.transform([LOCATIONS[0]])[0]

        # Combine features: scaled numerical + encoded categorical
        location_array = np.array([[float(location_encoded)]])
        final_input = np.hstack([scaled_numerical, location_array])

        # Predict (in log space)
        log_price = MODEL.predict(final_input)[0]

        # Back-transform to original scale
        price_inr = np.expm1(log_price)
        price_crores = price_inr / 10000000

        # Calculate metrics
        price_per_sqft = price_inr / area
        price_per_bedroom = price_crores / bedrooms

        # Format output
        result = f"""
╔════════════════════════════════════════════════════════╗
║           MUMBAI PROPERTY PRICE PREDICTION             ║
╚════════════════════════════════════════════════════════╝

PROPERTY DETAILS
────────────────────────────────────────────────────────
Location:                {location}
Area:                    {area:,.0f} sq.ft
Bedrooms:                {int(bedrooms)}

PREDICTED PRICE
────────────────────────────────────────────────────────
Total Price:             ₹{price_crores:.2f} Crores
                         ₹{price_inr:,.0f}

PRICE BREAKDOWN
────────────────────────────────────────────────────────
Price per sq.ft:         ₹{price_per_sqft:,.0f}
Price per bedroom:       ₹{price_per_bedroom:.2f} Crores

AMENITIES INCLUDED
────────────────────────────────────────────────────────
Sports Facility:         {sports}
Club House:              {club}
Air Conditioning:        {ac}
Gymnasium:               {gym}
Swimming Pool:           {pool}
24x7 Security:           {security}

────────────────────────────────────────────────────────
Model: {type(MODEL).__name__}
Prediction confidence: Based on {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)} features
        """

        return result

    except Exception as e:
        return f"""
ERROR IN PREDICTION
────────────────────────────────────────────────────────
{str(e)}

Please check:
1. All input values are valid
2. Model files are correctly loaded
3. Feature preprocessing matches training

If the error persists, please retrain the model.
        """


# ================================
# 4. CREATE GRADIO INTERFACE
# ================================

with gr.Blocks(theme=gr.themes.Soft(), title="Mumbai Housing Price Predictor") as app:
    gr.Markdown("""
    # 🏙️ Mumbai Housing Price Predictor

    Get accurate price predictions for residential properties in Mumbai using advanced machine learning.
    This model analyzes **41 features** including location, area, amenities, and more.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Property Specifications")

            area_input = gr.Slider(
                minimum=200,
                maximum=8000,
                value=1000,
                step=50,
                label="Area (Square Feet)",
                info="Enter the total area of the property"
            )

            bedrooms_input = gr.Slider(
                minimum=1,
                maximum=6,
                value=2,
                step=1,
                label="Number of Bedrooms",
                info="Choose between 1-6 bedrooms"
            )

            location_input = gr.Dropdown(
                choices=LOCATIONS,
                value=LOCATIONS[0],
                label="Location in Mumbai",
                info="Select the property location"
            )

            gr.Markdown("### ✨ Key Amenities")

            with gr.Row():
                sports_input = gr.Radio(
                    choices=['Yes', 'No'],
                    value='Yes',
                    label="Sports Facility"
                )
                club_input = gr.Radio(
                    choices=['Yes', 'No'],
                    value='Yes',
                    label="Club House"
                )

            with gr.Row():
                ac_input = gr.Radio(
                    choices=['Yes', 'No'],
                    value='No',
                    label="Air Conditioning"
                )
                gym_input = gr.Radio(
                    choices=['Yes', 'No'],
                    value='Yes',
                    label="Gymnasium"
                )

            with gr.Row():
                pool_input = gr.Radio(
                    choices=['Yes', 'No'],
                    value='Yes',
                    label="Swimming Pool"
                )
                security_input = gr.Radio(
                    choices=['Yes', 'No'],
                    value='Yes',
                    label="24x7 Security"
                )

            predict_btn = gr.Button("🔮 Predict Price", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 💰 Price Prediction")

            output = gr.Textbox(
                label="Prediction Result",
                lines=30,
                show_copy_button=True,
                placeholder="Enter property details and click 'Predict Price' to see results..."
            )

    # Examples section
    gr.Markdown("### 📊 Try These Examples")
    gr.Examples(
        examples=[
            [1000, 2, LOCATIONS[0], 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
            [1500, 3, LOCATIONS[1] if len(LOCATIONS) > 1 else LOCATIONS[0], 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
            [2500, 4, LOCATIONS[2] if len(LOCATIONS) > 2 else LOCATIONS[0], 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
            [800, 2, LOCATIONS[3] if len(LOCATIONS) > 3 else LOCATIONS[0], 'Yes', 'No', 'No', 'Yes', 'No', 'Yes'],
        ],
        inputs=[area_input, bedrooms_input, location_input, sports_input, club_input,
                ac_input, gym_input, pool_input, security_input],
        outputs=output,
        cache_examples=False
    )

    # Footer
    gr.Markdown("""
    ---
    ### ℹ️ About This Model

    - **Algorithm**: Advanced Ensemble Model (XGBoost/Gradient Boosting/Random Forest)
    - **Training Data**: Mumbai housing market data with 40+ features
    - **Preprocessing**: Log transformation, robust scaling, feature engineering
    - **Features**: Area, location, amenities, and engineered features (price per sqft, area per bedroom)

    **Note**: Predictions are estimates based on historical data and should be used as a reference only.
    """)

    # Connect button to prediction function
    predict_btn.click(
        fn=predict_house_price,
        inputs=[area_input, bedrooms_input, location_input, sports_input, club_input,
                ac_input, gym_input, pool_input, security_input],
        outputs=output
    )

# ================================
# 5. LAUNCH APP
# ================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LAUNCHING GRADIO APPLICATION")
    print("=" * 70)
    print("\nServer starting...")
    print("Access the app at the URL shown below\n")

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )