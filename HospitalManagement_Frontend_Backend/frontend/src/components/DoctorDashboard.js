// components/DoctorDashboard.js
import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import './DoctorDashboard.css';

function DoctorDashboard() {
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [uploadType, setUploadType] = useState(null);
  const [patientHistory, setPatientHistory] = useState(null);
  const [searchAadhar, setSearchAadhar] = useState('');
  const { user } = useAuth();

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    try {
      const response = await fetch('http://localhost:5001/patients', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      const data = await response.json();
      setPatients(data);
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const fetchPatientHistory = async (patientId) => {
    try {
      const response = await fetch(`http://localhost:5001/patient/history/${patientId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      const data = await response.json();
      setPatientHistory(data);
    } catch (error) {
      console.error('Error fetching patient history:', error);
    }
  };
  function StatBadge({ label, value, unit }) {
    return (
      <div className="stat-badge">
        <span className="stat-label">{label}</span>
        <span className="stat-value">
          {value}
          {unit && <span className="stat-unit">{unit}</span>}
        </span>
      </div>
    );
  }
  const handlePatientSelect = (patient) => {
    setSelectedPatient(patient);
    fetchPatientHistory(patient.id);
    setUploadType(null);
  };

  const filterPatients = () => {
    if (!searchAadhar) return patients;
    return patients.filter(patient => 
      patient.aadhar.toLowerCase().includes(searchAadhar.toLowerCase())
    );
  };


  const handleCTScanUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch(`http://127.0.0.1:5001/upload/ctscan/${selectedPatient.id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: formData,
      });
      if (response.ok) {
        alert('CT Scan uploaded successfully');
        setUploadType(null);
      }
    } catch (error) {
      console.error('Error uploading CT scan:', error);
    }
  };

  const handleMetabolicReport = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
      age: parseInt(formData.get('age')),
      sex: formData.get('sex'),
      waist_circ: parseFloat(formData.get('waistCirc')),
      bmi: parseFloat(formData.get('bmi')),
      albuminuria: parseFloat(formData.get('albuminuria')),
      ur_alb_cr: parseFloat(formData.get('urAlbCr')),
      uric_acid: parseFloat(formData.get('uricAcid')),
      blood_glucose: parseFloat(formData.get('bloodGlucose')),
      hdl: parseFloat(formData.get('hdl')),
      triglycerides: parseFloat(formData.get('triglycerides')),
    };

    try {
      const response = await fetch(`http://127.0.0.1:5001/upload/metabolic/${selectedPatient.id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (response.ok) {
        alert('Metabolic report uploaded successfully');
        setUploadType(null);
      }
    } catch (error) {
      console.error('Error uploading metabolic report:', error);
    }
  };

  const handleOrganReport = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
      age: parseInt(formData.get('age')),
      bmi: parseFloat(formData.get('bmi')),
      blood_pressure: formData.get('bloodPressure'),
      prg: parseFloat(formData.get('prg')),
      pl: parseFloat(formData.get('pl')),
      sk: parseFloat(formData.get('sk')),
      ts: parseFloat(formData.get('ts')),
      bd2: parseFloat(formData.get('bd2')),
    };

    try {
      const response = await fetch(`http://127.0.0.1:5001/upload/organ/${selectedPatient.id}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      if (response.ok) {
        alert('Organ report uploaded successfully');
        setUploadType(null);
      }
    } catch (error) {
      console.error('Error uploading organ report:', error);
    }
  };
  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1 className="dashboard-title">Doctor Dashboard</h1>
      </div>
      
      <div className="dashboard-grid">
        {/* Patient Selection Panel */}
        <div className="patient-list">
          <h2>Select Patient</h2>
          <input
            type="text"
            className="patient-search"
            placeholder="Search by Aadhar number"
            value={searchAadhar}
            onChange={(e) => setSearchAadhar(e.target.value)}
          />
          <div>
            {filterPatients().map((patient) => (
              <div
                key={patient.id}
                className={`patient-item ${selectedPatient?.id === patient.id ? 'selected' : ''}`}
                onClick={() => handlePatientSelect(patient)}
              >
                <p>{patient.username}</p>
                <p>Aadhar: {patient.aadhar}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="main-content">
          {selectedPatient ? (
            <>
              <h2>{selectedPatient.username}</h2>
              <p>Aadhar: {selectedPatient.aadhar}</p>

              <div className="upload-options">
                <button
                  className={`upload-button ${uploadType === 'ctscan' ? 'active' : ''}`}
                  onClick={() => setUploadType('ctscan')}
                >
                  Upload CT Scan
                </button>
                <button
                  className={`upload-button ${uploadType === 'metabolic' ? 'active' : ''}`}
                  onClick={() => setUploadType('metabolic')}
                >
                  Metabolic Report
                </button>
                <button
                  className={`upload-button ${uploadType === 'organ' ? 'active' : ''}`}
                  onClick={() => setUploadType('organ')}
                >
                  Organ Report
                </button>
              </div>

              {uploadType && (
                <div className="upload-form">
                  <h3 className="text-lg font-semibold mb-4">
                    {uploadType === 'ctscan' ? 'Upload CT Scan' :
                     uploadType === 'metabolic' ? 'Submit Metabolic Report' :
                     'Submit Organ Report'}
                  </h3>
                  
                  {/* Keep your existing form components here */}
                  {uploadType === 'ctscan' && (
                    <div className="mt-4">
                      <input
                        type="file"
                        accept=".jpg,.jpeg,.png"
                        onChange={handleCTScanUpload}
                        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                      />
                    </div>
                  )}

                  {uploadType === 'metabolic' && (
                    <form onSubmit={handleMetabolicReport} className="space-y-4">
                      <input type="number" name="age" placeholder="Age" required className="block w-full p-2 border rounded" />
                    <select name="sex" required className="block w-full p-2 border rounded">
                      <option value="">Select Sex</option>
                      <option value="M">Male</option>
                      <option value="F">Female</option>
                    </select>
                    <input type="number" step="0.1" name="waistCirc" placeholder="Waist Circumference" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="bmi" placeholder="BMI" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="albuminuria" placeholder="Albuminuria" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="urAlbCr" placeholder="UrAlbCr" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="uricAcid" placeholder="Uric Acid" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="bloodGlucose" placeholder="Blood Glucose" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="hdl" placeholder="HDL" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="triglycerides" placeholder="Triglycerides" required className="block w-full p-2 border rounded" />
                    <button type="submit" className="w-full py-2 px-4 bg-blue-600 text-white rounded hover:bg-blue-700">
                      Submit Report
                    </button>
                    </form>
                  )}

                  {uploadType === 'organ' && (
                    <form onSubmit={handleOrganReport} className="space-y-4">
                       <input type="number" name="age" placeholder="Age" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="bmi" placeholder="BMI" required className="block w-full p-2 border rounded" />
                    <input type="text" name="bloodPressure" placeholder="Blood Pressure" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="prg" placeholder="PRG" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="pl" placeholder="PL" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="sk" placeholder="SK" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="ts" placeholder="TS" required className="block w-full p-2 border rounded" />
                    <input type="number" step="0.1" name="bd2" placeholder="BD2" required className="block w-full p-2 border rounded" />
                    <button type="submit" className="w-full py-2 px-4 bg-blue-600 text-white rounded hover:bg-blue-700">
                      Submit Report
                    </button>
                    </form>
                  )}
                </div>
              )}

{patientHistory && (
  <div className="history-section">
    <div className="history-grid">
      {patientHistory.ct_scans.length > 0 && (
        <div className="history-category">
          <h4 className="section-subtitle">CT Scans ({patientHistory.ct_scans.length})</h4>
          <div className="ct-scan-grid">
            {patientHistory.ct_scans.map((scan) => (
              <div key={scan.id} className="scan-card">
                <img
                  src={`http://localhost:5001/uploads/${scan.image_path}`}
                  alt="CT Scan"
                  className="scan-image"
                />
                <div className="scan-meta">
                  <p className="text-sm text-gray-600">
                    {new Date(scan.upload_date).toLocaleDateString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {patientHistory.metabolic_reports.length > 0 && (
        <div className="history-category">
          <h4 className="section-subtitle">Metabolic Reports ({patientHistory.metabolic_reports.length})</h4>
          <div className="report-grid">
            {patientHistory.metabolic_reports.map((report) => (
              <div key={report.id} className="report-card">
                <div className="report-content">
                  <div className="report-meta">
                    <span className="report-age">{report.age} years</span>
                    <span className="report-sex">{report.sex}</span>
                  </div>
                  <div className="report-stats">
                    <StatBadge label="BMI" value={report.bmi} />
                    <StatBadge label="Glucose" value={report.blood_glucose} unit="mg/dL" />
                  </div>
                  <p className="report-date">
                    {new Date(report.upload_date).toLocaleDateString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {patientHistory.organ_reports.length > 0 && (
        <div className="history-category">
          <h4 className="section-subtitle">Organ Reports ({patientHistory.organ_reports.length})</h4>
          <div className="report-grid">
            {patientHistory.organ_reports.map((report) => (
              <div key={report.id} className="report-card">
                <div className="report-content">
                  <h5 className="report-title">Organ Health Summary</h5>
                  <div className="report-stats">
                    <StatBadge label="Blood Pressure: " value={report.blood_pressure} />
                    <StatBadge label="BMI: " value={report.bmi} />
                  </div>
                  <p className="report-date">
                  <StatBadge label="Date: " value={new Date(report.upload_date).toLocaleDateString()}/>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  </div>
)}
            </>
          ) : (
            <div className="empty-state">
              <p>Select a patient to view details and upload reports</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DoctorDashboard;