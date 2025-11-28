// ModelOutputs.js
import React from 'react';
import './ModelOutputs.css';
import outputData from "./output.json";

const ModelOutputs = () => {
  const {
    heart_disease,
    metabolic,
    organ,
    insurance,
    patient_info,
    risk_assessment_score,
  } = outputData;

  const getRiskColor = (score) => {
    if (score < 30) return '#22c55e';
    if (score < 60) return '#eab308';
    return '#ef4444';
  };

  const ProgressBar = ({ value, color }) => (
    <div className="progress-container">
      <div
        className="progress-bar"
        style={{
          width: `${Math.min(100, Math.max(0, value))}%`,
          backgroundColor: color || '#3b82f6'
        }}
      />
    </div>
  );

  return (
    <div className="dashboard">
      {/* Patient Info Card */}
      <div className="card">
        <div className="card-header">
          <h2>Patient Information</h2>
          <span className="badge">ID: {patient_info.id}</span>
        </div>
        <div className="card-content">
          <div className="info-grid">
            <div>
              <p className="label">Username</p>
              <p className="value">{patient_info.username}</p>
            </div>
            <div>
              <p className="label">Aadhar</p>
              <p className="value">{patient_info.aadhar}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Assessment Card */}
      <div className="card">
        <div className="card-header">
          <h2>Risk Assessment</h2>
        </div>
        <div className="card-content">
          <div className="risk-score">
            <div className="score-header">
              <span>Overall Risk Score</span>
              <span style={{ color: getRiskColor(risk_assessment_score) }}>
                {risk_assessment_score.toFixed(1)}%
              </span>
            </div>
            <ProgressBar
              value={risk_assessment_score}
              color={getRiskColor(risk_assessment_score)}
            />
          </div>
        </div>
      </div>

      {/* Heart Disease Analysis */}
      <div className="card">
        <div className="card-header">
          <h2>Heart Disease Analysis</h2>
        </div>
        <div className="card-content">
          <div className="analysis-content">
            <div className="metric">
              <span>Predicted Class</span>
              <span className="badge">{heart_disease.predicted_class}</span>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>Confidence</span>
                <span>{heart_disease.confidence.toFixed(2)}%</span>
              </div>
              <ProgressBar value={heart_disease.confidence} />
            </div>
            <div className="probabilities">
              {Object.entries(heart_disease.class_probabilities).map(([key, value]) => (
                <div key={key} className="probability-item">
                  <div className="metric-header">
                    <span>{key.replace(/\./g, ' ')}</span>
                    <span>{value.toFixed(2)}%</span>
                  </div>
                  <ProgressBar value={value} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Analysis Grid */}
      <div className="analysis-grid">
        {/* Metabolic Analysis */}
        <div className="card">
          <div className="card-header">
            <h2>Metabolic Analysis</h2>
          </div>
          <div className="card-content">
            <div className="metric">
              <span>Predicted Class</span>
              <span className="badge">{metabolic.predicted_class}</span>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>Probability</span>
                <span>{(metabolic.probability * 100).toFixed(2)}%</span>
              </div>
              <ProgressBar value={metabolic.probability * 100} color="#a855f7" />
            </div>
          </div>
        </div>

        {/* Organ Analysis */}
        <div className="card">
          <div className="card-header">
            <h2>Organ Analysis</h2>
          </div>
          <div className="card-content">
            <div className="metric">
              <span>Predicted Class</span>
              <span className="badge">{organ.predicted_class}</span>
            </div>
            <div className="metric">
              <div className="metric-header">
                <span>Probability</span>
                <span>{(organ.probability * 100).toFixed(2)}%</span>
              </div>
              <ProgressBar value={organ.probability * 100} color="#22c55e" />
            </div>
          </div>
        </div>

        {/* Insurance Prediction */}
        <div className="card">
          <div className="card-header">
            <h2>Insurance Prediction</h2>
          </div>
          <div className="card-content">
            <div className="prediction">
              <span className="label">Predicted Charge</span>
              <p className="charge">${insurance.predicted_charge.toFixed(2)}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelOutputs;
