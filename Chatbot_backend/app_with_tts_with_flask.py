from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import requests
import asyncio
import edge_tts
import tempfile
from dotenv import load_dotenv
import base64

app = Flask(__name__)
load_dotenv()

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
VOICE = "en-US-ChristopherNeural"

MEDICAL_RECORD_TEMPLATE = {
    "name": "",
    "age": "",
    "gender": "",
    "height": "",
    "weight": "",
    "blood_pressure": "",
    "temperature": "",
    "cholesterol_levels": "",
    "blood_sugar_levels": "",
    "drinking_habit": "",
    "smoking_habit": "",
}

# New key added for storing patient history
# (This key is not required to be in the template, but will be added once the name is provided.)

FIELD_DESCRIPTIONS = {
    "name": "full name",
    "age": "age in years",
    "gender": "gender (Male/Female/Other)",
    "height": "height in cm",
    "weight": "weight in kg",
    "blood_pressure": "blood pressure (e.g., 120/80)",
    "temperature": "body temperature in Celsius",
    "cholesterol_levels": "cholesterol levels",
    "blood_sugar_levels": "blood sugar levels",
    "drinking_habit": "drinking habits",
    "smoking_habit": "smoking habits",
}

def get_groq_response(messages, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def validate_answer(field, value):
    messages = [
        {"role": "system", "content": """You are a medical data validator. You must respond with EXACTLY one of these three options:
1. If the answer is valid, respond with only the word 'VALID'
2. if the user dont want to respond to your question ,politely tell them that it is needed for their aid and medical welfare and reframe the question and ask again. start your response with "I understand"
3. If the answer is invalid, respond with a short error message starting with 'Invalid:'"""},
        {"role": "user", "content": f"Validate this {field}: {value}"}
    ]

    response = get_groq_response(messages, temperature=0.1).strip()

    if response == "VALID":
        return True, "VALID"
    elif response.startswith("I understand"):
        return False, response
    elif not response.startswith("Invalid:"):
        return False, "Invalid: Please provide a valid response."
    return False, response

def generate_next_question(current_record):
    # Check fields in explicit order
    desired_order = ["name",
                     "age",
                     "gender",
                     "height",
                     "weight",
                     "blood_pressure",
                     "temperature",
                     "cholesterol_levels",
                     "blood_sugar_levels",
                     "drinking_habit",
                     "smoking_habit",]
    for field in desired_order:
        if not current_record.get(field):
            current_field = field
            break
    else:
        return None  # All fields filled

    description = FIELD_DESCRIPTIONS.get(current_field, current_field)

    messages = [
        {"role": "system", "content": "Ask one clear question to collect the specified information."},
        {"role": "user", "content": f"Ask for the patient's {description}"}
    ]

    return current_field, get_groq_response(messages)


import json
import subprocess

import time
def generate_summary(complete_record):
    # Write the complete record to a JSON file
    with open("complete_record.json", "w") as f:
        json.dump(complete_record, f, indent=2)

    # Run eval.py which is in the same directory
    subprocess.run(["python", "eval.py"], check=True)
    # time.sleep(3)
    subprocess.run(["python", "report.py"], check=True)
    # time.sleep(3)
    # subprocess.run(["python", "eval2.py"], check=True)

    # Prepare messages for the groq response
    messages = [
        {
            "role": "system",
            "content": ("You are a medical assistant. Generate a comprehensive summary "
                        "of the patient's medical record. Include key observations and potential concerns.")
        },
        {
            "role": "user",
            "content": f"Complete medical record: {json.dumps(complete_record)}"
        }
    ]

    return get_groq_response(messages)


def get_patient_history(username):
    """Calls the external endpoint using the provided username (first name) and returns the JSON response."""
    url = f"http://127.0.0.1:5001/patient/history/by-username/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching patient history: {str(e)}")
        return None

async def text_to_speech(text):
    """Convert text to speech using edge-tts and return the audio data"""
    communicate = edge_tts.Communicate(text, VOICE)

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    temp_file.close()  # Close the file handle immediately

    try:
        # Save the audio to the temporary file
        await communicate.save(temp_file.name)

        # Read the audio data
        with open(temp_file.name, 'rb') as audio_file:
            audio_data = audio_file.read()

        return base64.b64encode(audio_data).decode('utf-8')

    finally:
        # Clean up the temporary file using a try-except block
        try:
            os.unlink(temp_file.name)
        except PermissionError:
            # If we can't delete now, let the OS clean it up later
            pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_conversation():
    return jsonify({
        'medical_record': MEDICAL_RECORD_TEMPLATE.copy(),
        'current_field': None,
        'chat_history': []
    })

@app.route('/api/next_question', methods=['POST'])
def next_question():
    data = request.json
    medical_record = data.get('medical_record', {})

    current_field, question = generate_next_question(medical_record)

    try:
        audio_data = asyncio.run(text_to_speech(question))
        return jsonify({
            'current_field': current_field,
            'question': question,
            'audio_data': audio_data
        })
    except Exception as e:
        # Handle any TTS errors gracefully
        print(f"TTS Error: {str(e)}")
        return jsonify({
            'current_field': current_field,
            'question': question,
            'audio_data': None  # Client will handle missing audio
        })

@app.route('/api/submit_answer', methods=['POST'])
def submit_answer():
    data = request.json
    current_field = data.get('current_field')
    answer = data.get('answer')
    medical_record = data.get('medical_record', {})

    is_valid, message = validate_answer(current_field, answer)

    try:
        if is_valid:
            # Store the answer
            medical_record[current_field] = answer

            # If the field is "name", extract the first name and get patient history
            if current_field == "name":
                first_name = answer.split()[0]  # extract first name from full name
                patient_history = get_patient_history(first_name.lower())
                with open('patient_history.json', 'w') as file:
                    json.dump(patient_history, file)
                # Store the returned JSON for future use
                medical_record["patient_history"] = patient_history

            # Check if all fields are filled
            if all(medical_record.get(field) for field in MEDICAL_RECORD_TEMPLATE.keys()):
                summary = generate_summary(medical_record)
                audio_data = asyncio.run(text_to_speech(summary))
                return jsonify({
                    'status': 'complete',
                    'medical_record': medical_record,
                    'summary': summary,
                    'audio_data': audio_data
                })

            # Get next question
            next_field, question = generate_next_question(medical_record)
            audio_data = asyncio.run(text_to_speech(question))

            return jsonify({
                'status': 'success',
                'medical_record': medical_record,
                'current_field': next_field,
                'question': question,
                'audio_data': audio_data
            })
        else:
            audio_data = asyncio.run(text_to_speech(message))
            return jsonify({
                'status': 'error',
                'message': message,
                'audio_data': audio_data
            })
    except Exception as e:
        print(f"Error in submit_answer: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An error occurred while processing your request.',
            'audio_data': None
        })

if __name__ == '__main__':
    app.run(debug=True)
