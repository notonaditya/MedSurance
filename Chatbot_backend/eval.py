import json
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Custom JSON encoder for NumPy types
def numpy_encoder(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# -------------------------------
# 1. PyTorch Heart Disease (CT Scan) Model
# -------------------------------
class HeartDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(HeartDiseaseClassifier, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.efficient_net(x)


def load_heart_model(checkpoint_path, num_classes):
    model = HeartDiseaseClassifier(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_heart_image(model, image_path, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probability, predicted_idx = torch.max(probabilities, 1)
    predicted_class = class_names[predicted_idx.item()]
    confidence = probability.item() * 100
    all_probabilities = probabilities[0].cpu().numpy() * 100
    class_probabilities = {cls: float(prob) for cls, prob in zip(class_names, all_probabilities)}
    return predicted_class, confidence, class_probabilities


# -------------------------------
# 2. Metabolic Syndrome Predictor (Keras)
# -------------------------------
def load_metabolic_model(model_path):
    return keras.models.load_model(model_path)


class DummyScaler:
    def transform(self, X):
        return X


def predict_metabolic(model, scaler, report):
    sex = 1 if report['sex'].upper() == 'M' else 0
    features = [
        report['age'],
        sex,
        report['waist_circ'],
        report['bmi'],
        report['albuminuria'],
        report['ur_alb_cr'],
        report['uric_acid'],
        report['blood_glucose'],
        report['hdl'],
        report['triglycerides']
    ]
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_prob = model.predict(X_scaled)[0][0]
    predicted_class = 1 if pred_prob >= 0.5 else 0
    return predicted_class, float(pred_prob)


# -------------------------------
# 3. Organ Predictor (Keras)
# -------------------------------
def load_organ_model(model_path):
    return keras.models.load_model(model_path)


def predict_organ(model, scaler, report):
    try:
        blood_pressure = float(report['blood_pressure'])
    except ValueError:
        blood_pressure = 0.0
    features = [
        report['age'],
        report['bd2'],
        blood_pressure,
        report['bmi'],
        report['pl'],
        report['prg'],
        report['sk'],
        report['ts']
    ]
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_prob = model.predict(X_scaled)[0][0]
    predicted_class = 1 if pred_prob >= 0.5 else 0
    return predicted_class, float(pred_prob)


# -------------------------------
# 4. Insurance Charges Predictor (PyTorch)
# -------------------------------
def predict_insurance(model_path, complete_record, metabolic_bmi):
    age = float(complete_record["age"])
    gender = complete_record["gender"].strip().lower()
    sex_encoded = 1 if gender == "male" else 0
    bmi = float(metabolic_bmi)
    children = float(2)
    smoking_habit = complete_record["smoking_habit"].strip().lower()
    smoker_encoded = 1 if "yes" in smoking_habit else 0
    region_encoded = 0  # default value
    features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    scaler = DummyScaler()
    features_scaled = scaler.transform(features)

    class InsuranceNN(nn.Module):
        def __init__(self, input_dim):
            super(InsuranceNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(64, 32),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.model(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = features_scaled.shape[1]
    insurance_model = InsuranceNN(input_dim).to(device)
    insurance_model.load_state_dict(torch.load(model_path, map_location=device))
    insurance_model.eval()
    X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = insurance_model(X_tensor)
    y_pred_np = y_pred.cpu().numpy()[0][0]
    return y_pred_np


# -------------------------------
# Risk Assessment Calculation
# -------------------------------
def calculate_risk(output):
    # Extract the individual risk components:
    heart_confidence = output['heart_disease']['confidence']
    heart_class = output['heart_disease']['predicted_class']
    metabolic_prob = output['metabolic']['probability'] * 100  # convert to percentage
    organ_prob = output['organ']['probability'] * 100  # convert to percentage
    insurance_charge = output['insurance']['predicted_charge']

    # Assume that if heart predicted class is "normal", heart risk is 0
    heart_risk = heart_confidence if heart_class.lower() != "normal" else 0

    # For insurance risk, assume a maximum charge of 5 represents 100% risk
    insurance_risk = (insurance_charge / 5) * 100

    # Overall risk score is the average of these four factors
    risk_score = (heart_risk + metabolic_prob + organ_prob + insurance_risk) / 4
    return risk_score


# -------------------------------
# Main Inference Script
# -------------------------------
def main():
    output = {}

    # Load the patient history JSON (for heart, metabolic, and organ models)
    with open('patient_history.json', 'r') as f:
        patient_data = json.load(f)

    # ---- Heart Disease Inference ----
    ct_info = patient_data.get('ct_scans', [])[0]
    image_path = ct_info.get('image_path')
    heart_checkpoint_path = os.path.join('models', 'heart_disease.pth')
    heart_class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
    heart_model = load_heart_model(heart_checkpoint_path, len(heart_class_names))
    heart_predicted_class, heart_confidence, heart_class_probabilities = predict_heart_image(
        heart_model, image_path, heart_class_names)
    output['heart_disease'] = {
        'predicted_class': heart_predicted_class,
        'confidence': heart_confidence,
        'class_probabilities': heart_class_probabilities
    }

    # ---- Metabolic Syndrome Inference ----
    metabolic_report = patient_data.get('metabolic_reports', [])[0]
    metabolic_model_path = os.path.join('models', 'metabolic_syn.h5')
    metabolic_model = load_metabolic_model(metabolic_model_path)
    metabolic_scaler = DummyScaler()
    metabolic_predicted_class, metabolic_probability = predict_metabolic(
        metabolic_model, metabolic_scaler, metabolic_report)
    output['metabolic'] = {
        'predicted_class': metabolic_predicted_class,
        'probability': metabolic_probability
    }

    # ---- Organ Predictor Inference ----
    organ_report = patient_data.get('organ_reports', [])[0]
    organ_model_path = os.path.join('models', 'organ_model.h5')
    organ_model = load_organ_model(organ_model_path)
    organ_scaler = DummyScaler()
    organ_predicted_class, organ_probability = predict_organ(
        organ_model, organ_scaler, organ_report)
    output['organ'] = {
        'predicted_class': organ_predicted_class,
        'probability': organ_probability
    }

    # ---- Insurance Charges Prediction Inference ----
    with open('complete_record.json', 'r') as f:
        complete_record = json.load(f)
    metabolic_bmi = metabolic_report.get('bmi')
    insurance_model_path = os.path.join('models', 'insurance_model_v2.pth')
    insurance_prediction = predict_insurance(insurance_model_path, complete_record, metabolic_bmi)
    output['insurance'] = {
        'predicted_charge': insurance_prediction
    }

    # Include patient info from patient_history.json for reference
    output['patient_info'] = patient_data.get('patient_info', {})

    # Calculate overall risk assessment score based on our heuristic
    risk_score = calculate_risk(output)
    output['risk_assessment_score'] = risk_score

    # Save the combined inference results into a new JSON file
    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4, default=numpy_encoder)

    print("Inference completed. Output saved to 'output.json'.")
    print(f"Calculated Risk Assessment Score: {risk_score:.2f}%")


if __name__ == "__main__":
    main()
