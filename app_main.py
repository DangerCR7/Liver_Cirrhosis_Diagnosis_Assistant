import streamlit as st
import numpy as np
import json
import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
import nibabel as nib  # To load MRI images (.nii)
import random

# ========== Constants ==========

IMG_SIZE = (128, 128, 64)  # Target MRI image size
MRI_MODEL_PATH = 'models/liver_cirrhosis_mri_model.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DB_PATH = 'patients.db'

# ========== Load Models ==========

disease_model = load_model('models/disease_detection_model.h5')
stage_model = load_model('models/liver_stage_model.h5')

# Load food dataset
food_df = pd.read_csv('liver_cirrhosis_diet_dataset.csv')

# ========== CNN Model Definition ==========

class LiverCNN(nn.Module):
    def __init__(self):
        super(LiverCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self._initialize_fc1()
        self.fc1 = nn.Linear(self.fc1_input_features, 64)
        self.fc2 = nn.Linear(64, 2)  # Output: 0 or 1 (Healthy or Cirrhosis)

    def _initialize_fc1(self):
        dummy_input = torch.zeros(1, 1, *IMG_SIZE)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        self.fc1_input_features = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ========== Helper Functions ==========

def preprocess_volume(volume, target_shape):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, factors, order=1).astype(np.float32)
    resized -= resized.min()
    if resized.max() > 0:
        resized /= resized.max()
    return resized

def predict_from_mri(filepath, model):
    img = nib.load(filepath).get_fdata()
    img = preprocess_volume(img, IMG_SIZE)
    img = np.expand_dims(img, axis=(0, 1))  # shape: (1, 1, D, H, W)
    img_tensor = torch.tensor(img, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return int(pred.item())

def save_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
        name TEXT, age INTEGER, gender TEXT,
        clinical_prediction TEXT, stage TEXT,
        treatment TEXT, mri_prediction TEXT
    )''')

    if isinstance(data[5], dict):
        data = list(data)
        data[5] = json.dumps(data[5])
    cursor.execute('INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?)', tuple(data))
    conn.commit()
    conn.close()

def generate_30_day_meal_plan(stage):
    """Generate a 30-day meal plan based on cirrhosis stage"""
    if stage == "Early":
        stage_data = food_df[food_df['Simplified Cirrhosis Stage'] == 'Early']
        dietary_focus = "Balanced Nutrition"
    elif stage == "Progressive":
        stage_data = food_df[food_df['Simplified Cirrhosis Stage'] == 'Progressive']
        dietary_focus = "Low Sodium, Adequate Protein"
    else:  # Advanced
        stage_data = food_df[food_df['Simplified Cirrhosis Stage'] == 'Advanced']
        dietary_focus = "Very Low Sodium, Managed Protein"
    
    # Filter by dietary focus
    stage_data = stage_data[stage_data['Dietary Focus'] == dietary_focus]
    
    if len(stage_data) == 0:
        # Fallback if no exact match
        stage_data = food_df[food_df['Simplified Cirrhosis Stage'] == stage]
    
    # Get all unique food options
    breakfast_options = stage_data['Food Suggestion 1'].unique().tolist()
    lunch_options = stage_data['Food Suggestion 2'].unique().tolist()
    dinner_options = stage_data['Food Suggestion 3'].unique().tolist()
    
    # Create meal plan with variety
    meal_plan = []
    used_combinations = set()
    
    for day in range(1, 31):
        # Try to find a unique combination
        for _ in range(100):  # Limit attempts to avoid infinite loop
            breakfast = random.choice(breakfast_options)
            lunch = random.choice(lunch_options)
            dinner = random.choice(dinner_options)
            combo = (breakfast, lunch, dinner)
            
            if combo not in used_combinations or len(used_combinations) >= (len(breakfast_options) * len(lunch_options) * len(dinner_options)):
                used_combinations.add(combo)
                meal_plan.append({
                    'Day': f'Day {day}',
                    'Breakfast': breakfast,
                    'Lunch': lunch,
                    'Dinner': dinner
                })
                break
    
    return pd.DataFrame(meal_plan)

# ========== Streamlit App ==========

st.title("🩺 Liver Cirrhosis Diagnosis Assistant")
st.markdown("Upload a patient's MRI and clinical data to evaluate cirrhosis condition.")

# --- Patient Info ---
name = st.text_input("👤 Patient Name")
age = st.number_input("🎂 Age", min_value=1, max_value=120)
gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])

# --- MRI Upload ---
uploaded_file = st.file_uploader("🧠 Upload T1-weighted MRI (.nii)", type=['nii'])

# --- Clinical Features ---
st.subheader("📋 Clinical Features Input")

feature_fields = {
    "Status": str,
    "Drug": str,
    "Age": int,
    "Sex": str,
    "Ascites": str,
    "Hepatomegaly": str,
    "Spiders": str,
    "Edema": str,
    "Bilirubin": float,
    "Cholesterol": float,
    "Albumin": float,
    "Copper": float,
    "Alk_Phos": float,
    "SGOT": float,
    "Tryglicerides": float,
    "Platelets": float,
    "Prothrombin": float
}

categorical_map = {
    "Status": {"C": 0, "CL": 1, "D": 2},
    "Drug": {"Placebo": 0, "D-penicillamine": 1},
    "Sex": {"M": 0, "F": 1},
    "Ascites": {"N": 0, "Y": 1},
    "Hepatomegaly": {"N": 0, "Y": 1},
    "Spiders": {"N": 0, "Y": 1},
    "Edema": {"N": 0, "Y": 1}
}

user_inputs = []
for feature, dtype in feature_fields.items():
    if dtype == int or dtype == float:
        val = st.number_input(feature, step=1.0 if dtype == float else 1)
        user_inputs.append(val)
    else:
        options = list(categorical_map[feature].keys())
        selected = st.selectbox(feature, options)
        user_inputs.append(categorical_map[feature][selected])

# ========== Prediction Button ==========

if st.button("🔍 Predict"):
    if not uploaded_file or not name.strip():
        st.warning("Please complete all fields and upload the MRI file.")
    else:
        # --- Clinical Prediction ---
        X = np.array([user_inputs])
        disease_pred = int(np.round(disease_model.predict(X)[0][0]))

        if disease_pred == 1:
            stage_pred_num = int(round(stage_model.predict(X)[0][0]))
            # Map stage number to stage name
            stage_mapping = {0: "Early", 1: "Progressive", 2: "Advanced"}
            stage_pred = stage_mapping.get(stage_pred_num, "Early")
            treatment_pred = f"Recommended Treatment for {stage_pred} Stage"
        else:
            stage_pred = "N/A"
            treatment_pred = "N/A"

        # --- MRI Prediction ---
        temp_file_path = "temp_mri.nii"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        cnn_model = LiverCNN().to(DEVICE)
        cnn_model.load_state_dict(torch.load(MRI_MODEL_PATH, map_location=DEVICE))
        mri_pred = predict_from_mri(temp_file_path, cnn_model)
        os.remove(temp_file_path)

        # --- Display Output ---
        st.success("✅ Cirrhosis Prediction Complete")
        st.markdown(f"**👤 Patient:** {name} | **Age:** {age} | **Gender:** {gender}")
        st.markdown(f"**🧪 Clinical Prediction:** {'Cirrhosis' if disease_pred else 'Healthy'}")
        st.markdown(f"**🧠 MRI Prediction:** {'Cirrhosis' if mri_pred else 'Healthy'}")

        if disease_pred:
            st.markdown(f"**📊 Cirrhosis Stage:** {stage_pred} Stage")
            st.markdown(f"**💊 Recommended Treatment:** {treatment_pred}")

            # --- Food Plan Recommendation ---
            st.subheader("🍽️ 30-Day Personalized Meal Plan")
            
            # Generate meal plan
            meal_plan_df = generate_30_day_meal_plan(stage_pred)
            
            # Display as table
            st.dataframe(meal_plan_df, hide_index=True)
            
            # Provide download options
            csv = meal_plan_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{name}_30_day_meal_plan.csv",
                mime='text/csv'
            )
            
            # Text version for download
            text_version = "Day\tBreakfast\tLunch\tDinner\n"
            for _, row in meal_plan_df.iterrows():
                text_version += f"{row['Day']}\t{row['Breakfast']}\t{row['Lunch']}\t{row['Dinner']}\n"
            
            st.download_button(
                label="📥 Download as Text",
                data=text_version,
                file_name=f"{name}_30_day_meal_plan.txt",
                mime='text/plain'
            )

        # --- Save Results to DB ---
        save_to_db((
            name, age, gender,
            'Cirrhosis' if disease_pred else 'Healthy',
            f"{stage_pred} Stage" if disease_pred else "N/A",
            treatment_pred if disease_pred else "N/A",
            'Cirrhosis' if mri_pred else 'Healthy'
        ))

        # --- Final Verdict ---
        if disease_pred == mri_pred:
            st.info("🤝 Both models agree: High confidence in prediction.")
        else:
            st.warning("⚠️ Models disagree. Consider additional tests or doctor review.")