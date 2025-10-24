import gradio as gr
import joblib
import numpy as np
import pandas as pd
import warnings
import traceback

warnings.filterwarnings("ignore")

# ================================
# 1. GLOBAL ASSET LOADING
# ================================

# Define paths
MODEL_PATH = 'final_best_model.joblib'
SCALER_PATH = 'robust_scaler.joblib'
ENCODERS_PATH = 'label_encoders.joblib'
FEATURE_LIST_PATH = 'final_feature_list.joblib'
NUM_COLS_PATH = 'final_numerical_cols.joblib'

# Global variables
BEST_MODEL = None
SCALER = None
LOCATION_ENCODER = None
FINAL_FEATURE_LIST = []
FINAL_NUMERICAL_NAMES = []


def load_assets():
    """Loads the trained model, scaler, encoders, and feature list."""
    global BEST_MODEL, SCALER, LOCATION_ENCODER, FINAL_FEATURE_LIST, FINAL_NUMERICAL_NAMES

    try:
        BEST_MODEL = joblib.load(MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        encoders_loaded = joblib.load(ENCODERS_PATH)
        FINAL_FEATURE_LIST = joblib.load(FEATURE_LIST_PATH)
        FINAL_NUMERICAL_NAMES = joblib.load(NUM_COLS_PATH)

        print("✓ Model, scaler, and feature lists loaded")

        # ================================
        # FIX: Handle different encoder formats
        # ================================
        print("\n🔍 Checking encoder format...")
        print(f"Encoder type: {type(encoders_loaded)}")

        if isinstance(encoders_loaded, dict):
            # If it's a dictionary of encoders
            print("  Format: Dictionary of encoders")
            if 'Location' in encoders_loaded:
                LOCATION_ENCODER = encoders_loaded['Location']
                print("  ✓ Found 'Location' encoder in dictionary")
            else:
                # Try to find any LabelEncoder in the dict
                for key, value in encoders_loaded.items():
                    if hasattr(value, 'classes_'):
                        LOCATION_ENCODER = value
                        print(f"  ✓ Using encoder from key: '{key}'")
                        break
                else:
                    raise ValueError("No LabelEncoder found in dictionary")

        elif isinstance(encoders_loaded, str):
            # If it's a string (incorrectly saved)
            print("  ⚠️  WARNING: Encoder saved as string!")
            print(f"  Content: {encoders_loaded[:100]}...")

            # Try to extract location names from the string
            # This is a workaround - you should re-save properly
            print("\n  Attempting to reconstruct encoder from string...")

            # Create a new LabelEncoder with dummy data
            from sklearn.preprocessing import LabelEncoder
            LOCATION_ENCODER = LabelEncoder()

            # Try to parse locations from string or use defaults
            # You'll need to provide the actual location list here
            mumbai_locations = [
                'Andheri East', 'Andheri West', 'Bandra East', 'Bandra West',
                'Borivali East', 'Borivali West', 'Chembur', 'Dadar East',
                'Dadar West', 'Goregaon East', 'Goregaon West', 'Juhu',
                'Kandivali East', 'Kandivali West', 'Khar West', 'Kurla West',
                'Malad East', 'Malad West', 'Mira Road East', 'Mulund West',
                'Powai', 'Thane West', 'Versova', 'Vile Parle East',
                'Vile Parle West', 'Worli'
            ]

            LOCATION_ENCODER.fit(mumbai_locations)
            print(f"  ⚠️  Created fallback encoder with {len(mumbai_locations)} locations")
            print("  ⚠️  IMPORTANT: Re-save your encoder properly!")

        elif hasattr(encoders_loaded, 'classes_'):
            # If it's directly a LabelEncoder
            LOCATION_ENCODER = encoders_loaded
            print("  ✓ Format: Direct LabelEncoder object")
        else:
            raise TypeError(f"Unexpected encoder type: {type(encoders_loaded)}")

        # ================================
        # Validate encoder
        # ================================
        if not hasattr(LOCATION_ENCODER, 'classes_'):
            raise ValueError("Failed to load a valid LabelEncoder")

        print(f"\n✓ Location encoder ready")
        print(f"  Available locations: {len(LOCATION_ENCODER.classes_)}")
        print(f"  Sample locations: {list(LOCATION_ENCODER.classes_[:5])}")

        print(f"\n✓ Model expects {len(FINAL_FEATURE_LIST)} features")
        print(f"✓ Scaler expects {len(FINAL_NUMERICAL_NAMES)} numerical features")

        # Print first few features for debugging
        print(f"\nFirst 5 features expected by model: {FINAL_FEATURE_LIST[:5]}")
        print(f"Last 5 features expected by model: {FINAL_FEATURE_LIST[-5:]}")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Asset file not found: {e}")
        print("Please ensure all .joblib files are in the same directory.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR loading assets: {e}")
        traceback.print_exc()
        exit()


load_assets()

# ================================
# 2. MODEL CONFIGURATION
# ================================

# Default values for hidden features
RAW_FEATURE_DEFAULTS = {
    # User inputs (will be overwritten)
    'Area': 1000.0,
    'No. of Bedrooms': 2.0,
    'SportsFacility': 0.0,
    'ClubHouse': 0.0,
    'AC': 0.0,

    # Hidden features with sensible defaults
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
    'ATM': 0.0,
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
# 3. PREDICTION FUNCTION (FIXED)
# ================================

def predict_price(area_sqft, bedrooms, location_name, has_sports_facility, has_club_house, has_ac):
    """
    Takes user inputs, transforms them, and returns the predicted price.
    Enhanced with detailed error handling and debugging.
    """
    try:
        print("\n" + "=" * 60)
        print("🔍 PREDICTION DEBUG LOG")
        print("=" * 60)

        # ================================
        # STEP 1: Create Base Feature Dictionary
        # ================================
        print("\n[STEP 1] Creating base feature dictionary...")
        data = RAW_FEATURE_DEFAULTS.copy()

        # Update with user inputs (ensure float type)
        data['Area'] = float(area_sqft)
        data['No. of Bedrooms'] = float(bedrooms)
        data['SportsFacility'] = 1.0 if has_sports_facility == 'Yes' else 0.0
        data['ClubHouse'] = 1.0 if has_club_house == 'Yes' else 0.0
        data['AC'] = 1.0 if has_ac == 'Yes' else 0.0

        print(f"  User Inputs:")
        print(f"    Area: {data['Area']} sq.ft")
        print(f"    Bedrooms: {data['No. of Bedrooms']}")
        print(f"    Location: {location_name}")
        print(f"    Sports Facility: {has_sports_facility}")
        print(f"    Club House: {has_club_house}")
        print(f"    AC: {has_ac}")

        # ================================
        # STEP 2: Create DataFrame
        # ================================
        print("\n[STEP 2] Creating DataFrame...")
        input_df = pd.DataFrame([data])
        print(f"  Initial DataFrame shape: {input_df.shape}")
        print(f"  Columns: {list(input_df.columns)[:5]}... (showing first 5)")

        # ================================
        # STEP 3: Feature Engineering
        # ================================
        print("\n[STEP 3] Applying feature engineering...")

        # Store original area for calculations
        original_area = float(input_df['Area'].iloc[0])
        num_bedrooms = float(input_df['No. of Bedrooms'].iloc[0])

        print(f"  Original Area: {original_area}")
        print(f"  Bedrooms: {num_bedrooms}")

        # Validate bedrooms
        if num_bedrooms <= 0:
            return "❌ Error: Number of bedrooms must be greater than 0"

        # Create log transformations and engineered features
        input_df['log_Area'] = np.log1p(original_area)

        # CRITICAL FIX: Price per sqft should be a constant assumption during inference
        # Since we don't know the actual price, we use area as a proxy
        # OR use a typical Mumbai price per sqft (e.g., 15000-20000 per sqft)
        ASSUMED_PRICE_PER_SQFT = 18000.0  # Typical Mumbai price per sqft
        assumed_price = original_area * ASSUMED_PRICE_PER_SQFT
        input_df['log_price_per_sqft'] = np.log1p(ASSUMED_PRICE_PER_SQFT)

        # Area per bedroom
        area_per_bedroom = original_area / num_bedrooms
        input_df['log_area_per_bedroom'] = np.log1p(area_per_bedroom)

        print(f"  ✓ log_Area: {input_df['log_Area'].iloc[0]:.4f}")
        print(f"  ✓ log_price_per_sqft: {input_df['log_price_per_sqft'].iloc[0]:.4f}")
        print(f"  ✓ log_area_per_bedroom: {input_df['log_area_per_bedroom'].iloc[0]:.4f}")

        # Drop original Area (no longer needed)
        input_df = input_df.drop('Area', axis=1)

        print(f"  DataFrame shape after engineering: {input_df.shape}")

        # ================================
        # STEP 4: Prepare Features for Scaling
        # ================================
        print("\n[STEP 4] Preparing numerical features for scaling...")

        # Get the numerical columns (should be 40 features)
        try:
            numerical_features = input_df[FINAL_NUMERICAL_NAMES].values
            print(f"  ✓ Extracted {numerical_features.shape[1]} numerical features")
            print(f"  Feature names: {FINAL_NUMERICAL_NAMES[:3]}... (showing first 3)")
        except KeyError as e:
            missing_cols = set(FINAL_NUMERICAL_NAMES) - set(input_df.columns)
            print(f"  ❌ Missing columns: {missing_cols}")
            return f"❌ Error: Missing required columns: {missing_cols}"

        # ================================
        # STEP 5: Scale Numerical Features
        # ================================
        print("\n[STEP 5] Scaling numerical features...")
        scaled_numerical = SCALER.transform(numerical_features)
        print(f"  ✓ Scaled features shape: {scaled_numerical.shape}")
        print(f"  Sample scaled values: {scaled_numerical[0][:3]}... (first 3)")

        # ================================
        # STEP 6: Encode Location
        # ================================
        print("\n[STEP 6] Encoding location...")
        try:
            loc_encoded = LOCATION_ENCODER.transform([location_name])[0]
            print(f"  ✓ Location '{location_name}' encoded as: {loc_encoded}")
        except ValueError:
            print(f"  ⚠️  Location '{location_name}' not found in training data")
            print(f"  Using default location: {LOCATION_ENCODER.classes_[0]}")
            loc_encoded = LOCATION_ENCODER.transform([LOCATION_ENCODER.classes_[0]])[0]

        # Create location array with proper shape
        location_array = np.array([[float(loc_encoded)]])
        print(f"  Location array shape: {location_array.shape}")

        # ================================
        # STEP 7: Combine Features
        # ================================
        print("\n[STEP 7] Combining all features...")

        # Stack: [scaled_numerical (40 features)] + [location (1 feature)] = 41 features
        final_input = np.hstack([scaled_numerical, location_array])

        print(f"  ✓ Final input shape: {final_input.shape}")
        print(f"  Expected shape: (1, {len(FINAL_FEATURE_LIST)})")

        # Validate feature count
        if final_input.shape[1] != len(FINAL_FEATURE_LIST):
            error_msg = f"""
            ❌ FEATURE MISMATCH ERROR:
            Expected: {len(FINAL_FEATURE_LIST)} features
            Got: {final_input.shape[1]} features

            Breakdown:
            - Numerical features: {scaled_numerical.shape[1]}
            - Location feature: 1
            - Total: {final_input.shape[1]}

            Please check if the model was trained with the same feature set.
            """
            print(error_msg)
            return error_msg.strip()

        print("  ✓ Feature count matches!")

        # ================================
        # STEP 8: Make Prediction
        # ================================
        print("\n[STEP 8] Making prediction...")

        log_price_pred = BEST_MODEL.predict(final_input)[0]
        print(f"  Log-transformed prediction: {log_price_pred:.4f}")

        # Inverse transform from log space
        price_inr = np.expm1(log_price_pred)
        price_crores = price_inr / 10000000

        print(f"  ✓ Predicted price: ₹{price_inr:,.0f}")
        print(f"  ✓ Predicted price: ₹{price_crores:.2f} Crores")

        # ================================
        # STEP 9: Format Output
        # ================================
        print("\n[STEP 9] Formatting output...")

        # Create detailed output
        output = f"""
╔══════════════════════════════════════════════════════════╗
║         MUMBAI PROPERTY PRICE PREDICTION                 ║
╚══════════════════════════════════════════════════════════╝

📍 Location: {location_name}
📐 Area: {original_area:,.0f} sq.ft
🛏️  Bedrooms: {int(num_bedrooms)}

💰 PREDICTED PRICE: ₹{price_crores:.2f} Crores
                    (₹{price_inr:,.0f})

📊 Price Breakdown:
   • Price per sq.ft: ₹{(price_inr / original_area):,.0f}
   • Price per bedroom: ₹{(price_crores / num_bedrooms):.2f} Crores

✨ Selected Amenities:
   • Sports Facility: {has_sports_facility}
   • Club House: {has_club_house}
   • Air Conditioning: {has_ac}

⚡ Model: XGBoost (Tuned)
📈 Features Used: {len(FINAL_FEATURE_LIST)}
        """

        print("=" * 60)
        print("✓ PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return output.strip()

    except Exception as e:
        error_msg = f"""
        ❌ PREDICTION ERROR:
        {str(e)}

        Full traceback:
        {traceback.format_exc()}

        Please check:
        1. All .joblib files are in the correct directory
        2. Model was trained with the same preprocessing pipeline
        3. Feature names match exactly (case-sensitive)
        4. Input values are within valid ranges
        """
        print(error_msg)
        return error_msg.strip()


# ================================
# 4. GRADIO INTERFACE
# ================================

print("\n" + "=" * 60)
print("🚀 SETTING UP GRADIO INTERFACE")
print("=" * 60)

# Get location choices
try:
    LOCATION_CHOICES = sorted(list(LOCATION_ENCODER.classes_))
    print(f"✓ Available locations: {len(LOCATION_CHOICES)}")
    print(f"  Sample locations: {LOCATION_CHOICES[:5]}")
except AttributeError:
    LOCATION_CHOICES = ['Andheri', 'Bandra', 'Borivali', 'Dadar', 'Khar']
    print("⚠️  Using default location list")

# Define interface components
interface_inputs = [
    gr.Slider(
        minimum=200,
        maximum=8000,
        value=1000,
        step=50,
        label="🏠 Area (Square Feet)",
        info="Typical range: 500-5000 sq.ft"
    ),
    gr.Slider(
        minimum=1,
        maximum=6,
        value=2,
        step=1,
        label="🛏️ Number of Bedrooms",
        info="Choose between 1-6 bedrooms"
    ),
    gr.Dropdown(
        choices=LOCATION_CHOICES,
        value=LOCATION_CHOICES[0],
        label="📍 Location in Mumbai",
        info="Select the property location"
    ),
    gr.Radio(
        choices=['Yes', 'No'],
        value='Yes',
        label="🏃 Sports Facility",
        info="Does the property have sports facilities?"
    ),
    gr.Radio(
        choices=['Yes', 'No'],
        value='Yes',
        label="🏊 Club House",
        info="Is there a club house in the complex?"
    ),
    gr.Radio(
        choices=['Yes', 'No'],
        value='No',
        label="❄️ Air Conditioning (AC)",
        info="Does the property include AC?"
    ),
]

# Create examples for quick testing
example_inputs = [
    [1000, 2, LOCATION_CHOICES[0], 'Yes', 'Yes', 'No'],
    [1500, 3, LOCATION_CHOICES[1] if len(LOCATION_CHOICES) > 1 else LOCATION_CHOICES[0], 'Yes', 'Yes', 'Yes'],
    [2500, 4, LOCATION_CHOICES[2] if len(LOCATION_CHOICES) > 2 else LOCATION_CHOICES[0], 'No', 'Yes', 'Yes'],
]

# Create interface
iface = gr.Interface(
    fn=predict_price,
    inputs=interface_inputs,
    outputs=gr.Textbox(
        label="Prediction Result",
        lines=20,
        show_copy_button=True
    ),
    title="🏙️ Mumbai Housing Price Predictor",
    description="""
    ### Predict property prices in Mumbai using advanced ML

    This model uses **XGBoost** trained on Mumbai housing data with **41 features** including:
    - Property specifications (area, bedrooms)
    - Location (encoded from 100+ Mumbai localities)
    - Amenities (sports, club house, AC, etc.)
    - Hidden features (security, parking, utilities)

    **Note:** The model was trained on log-transformed prices for better accuracy.
    """,
    examples=example_inputs,
    cache_examples=False,
    allow_flagging='never',
    theme=gr.themes.Soft(),
)

print("✓ Gradio interface created")

# ================================
# 5. LAUNCH
# ================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌐 LAUNCHING GRADIO APP")
    print("=" * 60)
    print("✓ Interface ready")
    print("✓ Model loaded and ready for predictions")
    print("\nAccess the app at the URL shown below:")
    print("-" * 60)

    iface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )