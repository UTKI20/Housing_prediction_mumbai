import joblib
import os

# Define the paths used in the saving step
FEATURE_LIST_PATH = 'final_feature_list.joblib'
NUM_COLS_PATH = 'final_numerical_cols.joblib'

print("--- Feature Order Check ---")

# 1. Load the full feature list (should contain 20 names)
try:
    FINAL_FEATURE_LIST = joblib.load(FEATURE_LIST_PATH)
    print(f"File loaded: {FEATURE_LIST_PATH}")
    print(f"Total Features (Model Input Size): {len(FINAL_FEATURE_LIST)}")
    print("-" * 30)

    # 2. Print the list to inspect the exact order
    print("Exact Feature Order (Index: Feature Name):")
    for i, feature in enumerate(FINAL_FEATURE_LIST):
        # Use simple formatting to show the index and name
        if i < len(FINAL_FEATURE_LIST) - 1:
             # This is the numerical/engineered part
            print(f"[{i:02d}]: {feature}")
        else:
            # This is the LabelEncoded Location part
            print(f"[{i:02d}]: {feature} (Location Code)")

except FileNotFoundError:
    print(f"\nERROR: One or more feature list files were not found.")
    print(f"Please ensure '{FEATURE_LIST_PATH}' and '{NUM_COLS_PATH}' are in the current directory.")
except Exception as e:
    print(f"\nAn unexpected error occurred during loading: {e}")

print("--------------------------")
