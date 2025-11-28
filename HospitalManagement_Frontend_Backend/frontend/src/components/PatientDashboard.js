// components/PatientDashboard.js
import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import './PatientDashboard.css';

function PatientDashboard() {
  const [history, setHistory] = useState(null);
  const { user } = useAuth();

  

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:5001/patient/history/${user.user_id}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      const data = await response.json();
      setHistory(data);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  return (
    <div className="patient-dashboard">
      <h1 className="dashboard-title">Patient Dashboard</h1>
      
      {history && (
        <div>
          {history.ct_scans.length > 0 && (
          <section className="report-section">
            <h2>CT Scans</h2>
            <div className="scan-grid">
              {history.ct_scans.map((scan) => (
                <div key={scan.id} className="scan-card">
                  <img
                    src={`http://127.0.0.1:5001/uploads/${scan.image_path}`}
                    alt="CT Scan"
                    className="scan-image"
                  />
                  <div className="scan-info">
                    <p>Upload Date: {scan.upload_date}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
          )}

{history.metabolic_reports.length > 0 && (

          <section className="report-section">
            <h2>Metabolic Reports</h2>
            {history.metabolic_reports.map((report) => (
              <div key={report.id} className="report-card">
                <p>Age: {report.age}</p>
                <p>Sex: {report.sex}</p>
                <p>BMI: {report.bmi}</p>
                <p>Upload Date: {report.upload_date}</p>
              </div>
            ))}
          </section>

)}

{history.organ_reports.length > 0 && (
          <section className="report-section">
            <h2>Organ Reports</h2>
            {history.organ_reports.map((report) => (
              <div key={report.id} className="report-card">
                <p>Age: {report.age}</p>
                <p>BMI: {report.bmi}</p>
                <p>Blood Pressure: {report.blood_pressure}</p>
                <p>Upload Date: {report.upload_date}</p>
              </div>
            ))}
          </section>
              )}
        </div>
      )}
    </div>
  );
}

export default PatientDashboard;