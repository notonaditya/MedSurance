# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta, timezone
import os
IST = timezone(timedelta(hours=5, minutes=30))
app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hospital.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'doctor' or 'patient'
    aadhar = db.Column(db.String(12), unique=True, nullable=False)

class CTScan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.now(IST))

class MetabolicReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer)
    sex = db.Column(db.String(10))
    waist_circ = db.Column(db.Float)
    bmi = db.Column(db.Float)
    albuminuria = db.Column(db.Float)
    ur_alb_cr = db.Column(db.Float)
    uric_acid = db.Column(db.Float)
    blood_glucose = db.Column(db.Float)
    hdl = db.Column(db.Float)
    triglycerides = db.Column(db.Float)
    upload_date = db.Column(db.DateTime, default=datetime.now(IST))

class OrganReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    blood_pressure = db.Column(db.String(20))
    prg = db.Column(db.Float)
    pl = db.Column(db.Float)
    sk = db.Column(db.Float)
    ts = db.Column(db.Float)
    bd2 = db.Column(db.Float)
    upload_date = db.Column(db.DateTime, default=datetime.now(IST))

with app.app_context():
    db.create_all()

def token_required(f):
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token.split()[1], app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['user_id']).first()
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    decorator.__name__ = f.__name__
    return decorator

# Add this new route to app.py
@app.route('/patient/history/by-username/<username>', methods=['GET'])
def get_patient_history_by_username(username):
    # Find the patient by username
    patient = User.query.filter_by(username=username, role='patient').first()

    if not patient:
        return jsonify({'message': 'Patient not found'}), 404

    # Check authorization

    # Reuse the existing get_patient_history logic
    ct_scans = CTScan.query.filter_by(patient_id=patient.id).all()
    metabolic_reports = MetabolicReport.query.filter_by(patient_id=patient.id).all()
    organ_reports = OrganReport.query.filter_by(patient_id=patient.id).all()

    current_directory = os.getcwd()
    BASE_IMAGE_PATH=os.path.join(current_directory, "uploads")


    return jsonify({
        'patient_info': {
            'id': patient.id,
            'username': patient.username,
            'aadhar': patient.aadhar
        },
        'ct_scans': [{
            'image_path': os.path.join(BASE_IMAGE_PATH, scan.image_path) if scan.image_path else None
        } for scan in ct_scans],
        'metabolic_reports': [{
            'age': report.age,
            'sex': report.sex,
            'waist_circ': report.waist_circ,
            'bmi': report.bmi,
            'albuminuria': report.albuminuria,
            'ur_alb_cr': report.ur_alb_cr,
            'uric_acid': report.uric_acid,
            'blood_glucose': report.blood_glucose,
            'hdl': report.hdl,
            'triglycerides': report.triglycerides,
        } for report in metabolic_reports],
        'organ_reports': [{
            'age': report.age,
            'bmi': report.bmi,
            'blood_pressure': report.blood_pressure,
            'prg': report.prg,
            'pl': report.pl,
            'sk': report.sk,
            'ts': report.ts,
            'bd2': report.bd2,
        } for report in organ_reports]
    })

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    hashed_password = generate_password_hash(data['password'])
    new_user = User(
        username=data['username'],
        password=hashed_password,
        role=data['role'],
        aadhar=data['aadhar']
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        return jsonify({
            'token': token,
            'role': user.role,
            'user_id': user.id
        })
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/patients', methods=['GET'])
@token_required
def get_patients(current_user):
    if current_user.role != 'doctor':
        return jsonify({'message': 'Unauthorized'}), 403
    patients = User.query.filter_by(role='patient').all()
    return jsonify([{
        'id': p.id,
        'username': p.username,
        'aadhar': p.aadhar
    } for p in patients])

@app.route('/upload/ctscan/<int:patient_id>', methods=['POST'])
@token_required
def upload_ctscan(current_user, patient_id):
    if current_user.role != 'doctor':
        return jsonify({'message': 'Unauthorized'}), 403
    
    if 'image' not in request.files:
        return jsonify({'message': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    if file:
        filename = f"{datetime.now().timestamp()}_{file.filename}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        new_scan = CTScan(
            patient_id=patient_id,
            doctor_id=current_user.id,
            image_path=filename
        )
        db.session.add(new_scan)
        db.session.commit()
        return jsonify({'message': 'CT Scan uploaded successfully'})

@app.route('/upload/metabolic/<int:patient_id>', methods=['POST'])
@token_required
def upload_metabolic(current_user, patient_id):
    if current_user.role != 'doctor':
        return jsonify({'message': 'Unauthorized'}), 403
    
    data = request.json
    new_report = MetabolicReport(
        patient_id=patient_id,
        doctor_id=current_user.id,
        **data
    )
    db.session.add(new_report)
    db.session.commit()
    return jsonify({'message': 'Metabolic report uploaded successfully'})

# Add these imports at the top of app.py
from flask import send_from_directory

# Add this route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload/organ/<int:patient_id>', methods=['POST'])
@token_required
def upload_organ(current_user, patient_id):
    if current_user.role != 'doctor':
        return jsonify({'message': 'Unauthorized'}), 403
    
    data = request.json
    new_report = OrganReport(
        patient_id=patient_id,
        doctor_id=current_user.id,
        **data
    )
    db.session.add(new_report)
    db.session.commit()
    return jsonify({'message': 'Organ report uploaded successfully'})

@app.route('/patient/history/<int:patient_id>', methods=['GET'])
@token_required
def get_patient_history(current_user, patient_id):
    if current_user.role != 'doctor' and current_user.id != patient_id:
        return jsonify({'message': 'Unauthorized'}), 403
    
    ct_scans = CTScan.query.filter_by(patient_id=patient_id).all()
    metabolic_reports = MetabolicReport.query.filter_by(patient_id=patient_id).all()
    organ_reports = OrganReport.query.filter_by(patient_id=patient_id).all()
    
    return jsonify({
        'ct_scans': [{
            'id': scan.id,
            'image_path': scan.image_path,
            'upload_date': scan.upload_date.strftime('%Y-%m-%d %H:%M:%S')
        } for scan in ct_scans],
        'metabolic_reports': [{
            'id': report.id,
            'age': report.age,
            'sex': report.sex,
            'waist_circ': report.waist_circ,
            'bmi': report.bmi,
            'upload_date': report.upload_date.strftime('%Y-%m-%d %H:%M:%S')
        } for report in metabolic_reports],
        'organ_reports': [{
            'id': report.id,
            'age': report.age,
            'bmi': report.bmi,
            'blood_pressure': report.blood_pressure,
            'upload_date': report.upload_date.strftime('%Y-%m-%d %H:%M:%S')
        } for report in organ_reports]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)