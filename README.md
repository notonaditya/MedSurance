# Virtual Medical Examiner Assessment (VMEA)

## Overview
VMEA is an AI-driven medical evaluation system designed for insurance companies to automate health assessments efficiently. It integrates an AI-powered 3D medical avatar, multiple health assessment models, and hospital data to provide structured, real-time evaluations.

## Features
- **AI-Powered 3D Medical Avatar** for virtual assessments
- **Multi-Model Health Analysis** for accuracy
- **Hospital Database Integration** for seamless medical data retrieval
- **Automated Risk Scoring** for underwriting
- **Secure & Compliant** with industry standards

## Tech Stack
### Frontend
- **Framework:** React.js
- **3D Rendering:** Three.js (for AI Medical Avatar Visualization)
- **Voice Integration:** Web Audio API & Speech-to-Text
- **Standards:** HTML5, CSS3, JavaScript

### Backend
- **Server Framework:** Python (Flask)
- **AI/ML Integration:** TensorFlow/PyTorch for health models & custom LLMs
- **Database:** MySQL (for hospital data integration)
- **Security:** JWT for authentication and secure data transmission

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3
- Node.js & npm
- Virtual Environment for Python (`venv` recommended)

### Backend Setup
1. Clone the repository:
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the chatbot backend:
   ```bash
   cd Chatbot_backend
   python3 app_with_tts_with_flask.py
   ```
   ```bash
   python3 eval2.py
   ```
   - This will start the Flask server at `http://127.0.0.1:5000`

4. Run the hospital management backend:
   ```bash
   cd HospitalManagement_Frontend_Backend/backend
   python3 app.py
   ```
   - This will start the hospital API at `http://127.0.0.1:5001`

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd Final_frontendwithavatar
   ```
2. Install dependencies:
   ```bash
   npm i
   ```
3. Start the frontend application:
   ```bash
   npm start
   ```
   - This will start the frontend, allowing you to interact with the chatbot UI.


### Running Hospital Management Frontend
1. Navigate to the frontend folder:
   ```bash
   cd HospitalManagement_Frontend_Backend/frontend
   ```
2. Install dependencies:
   ```bash
   npm i
   ```
3. Start the frontend:
   ```bash
   npm start
   ```
## How It Works
- The **chatbot backend** (`app_with_tts_with_flask.py`) powers the AI-driven medical assistant.
- The **hospital management backend** (`app.py`) handles hospital data and patient records.
- The **frontend application** provides an interactive UI for virtual health assessments.
- The AI models evaluate health conditions and generate reports for insurance underwriting.
