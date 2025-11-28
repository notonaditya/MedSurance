import json
from datetime import datetime
import webbrowser
import os
import base64

def generate_enhanced_medical_report(patient_data, results_data):
    """
    Generate a beautifully styled HTML medical report with charts and enhanced visuals
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Report - {patient_data.get('name', 'Patient')}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.0/chart.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
            
            :root {{
                --primary-color: #2196F3;
                --secondary-color: #1976D2;
                --accent-color: #BBDEFB;
                --text-color: #333;
                --light-gray: #f5f5f5;
                --success-color: #4CAF50;
                --warning-color: #FFC107;
                --danger-color: #F44336;
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Poppins', sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                padding: 2rem;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(120deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 2rem;
                position: relative;
                overflow: hidden;
            }}
            
            .header::after {{
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                bottom: 0;
                left: 0;
                background: linear-gradient(120deg, rgba(255,255,255,0.2), transparent);
                pointer-events: none;
            }}
            
            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                font-weight: 600;
            }}
            
            .section {{
                padding: 2rem;
                margin: 1rem;
                background: white;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                transition: transform 0.3s ease;
            }}
            
            .section:hover {{
                transform: translateY(-5px);
            }}
            
            .section h2 {{
                color: var(--primary-color);
                font-size: 1.8rem;
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid var(--accent-color);
            }}
            
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                margin: 2rem 0;
            }}
            
            .info-card {{
                background: var(--light-gray);
                padding: 1.5rem;
                border-radius: 10px;
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            
            .info-card i {{
                font-size: 2rem;
                color: var(--primary-color);
            }}
            
            .vital-sign {{
                text-align: center;
                padding: 1.5rem;
                background: white;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            
            .vital-sign h3 {{
                color: var(--primary-color);
                font-size: 1.2rem;
                margin-bottom: 0.5rem;
            }}
            
            .vital-sign .value {{
                font-size: 2rem;
                font-weight: 600;
                color: var(--secondary-color);
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
            }}
            
            th, td {{
                padding: 1rem;
                text-align: left;
                border-bottom: 1px solid var(--light-gray);
            }}
            
            th {{
                background: var(--primary-color);
                color: white;
                font-weight: 500;
            }}
            
            tr:hover {{
                background: var(--light-gray);
            }}
            
            .risk-indicator {{
                padding: 2rem;
                text-align: center;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
            }}
            
            .chart-container {{
                width: 100%;
                max-width: 600px;
                margin: 2rem auto;
                padding: 1rem;
                background: white;
                border-radius: 10px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            
            .button-container {{
                text-align: center;
                padding: 2rem;
            }}
            
            .download-btn {{
                background: var(--primary-color);
                color: white;
                padding: 1rem 2rem;
                border: none;
                border-radius: 50px;
                font-size: 1.1rem;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
            }}
            
            .download-btn:hover {{
                background: var(--secondary-color);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                
                .container {{
                    box-shadow: none;
                }}
                
                .download-btn {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-hospital-user"></i> Medical Report</h1>
                <p><i class="fas fa-calendar"></i> Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>

            <div class="section">
                <h2><i class="fas fa-user"></i> Patient Information</h2>
                <div class="grid-container">
                    <div class="info-card">
                        <i class="fas fa-user-circle"></i>
                        <div>
                            <h3>Name</h3>
                            <p>{patient_data.get('name', 'N/A')}</p>
                        </div>
                    </div>
                    <div class="info-card">
                        <i class="fas fa-birthday-cake"></i>
                        <div>
                            <h3>Age</h3>
                            <p>{patient_data.get('age', 'N/A')} years</p>
                        </div>
                    </div>
                    <div class="info-card">
                        <i class="fas fa-venus-mars"></i>
                        <div>
                            <h3>Gender</h3>
                            <p>{patient_data.get('gender', 'N/A')}</p>
                        </div>
                    </div>
                </div>

                <div class="grid-container">
                    <div class="vital-sign">
                        <h3><i class="fas fa-weight"></i> Weight</h3>
                        <div class="value">{patient_data.get('weight', 'N/A')} </div>
                    </div>
                    <div class="vital-sign">
                        <h3><i class="fas fa-ruler-vertical"></i> Height</h3>
                        <div class="value">{patient_data.get('height', 'N/A')}</div>
                    </div>
                    <div class="vital-sign">
                        <h3><i class="fas fa-heartbeat"></i> Blood Pressure</h3>
                        <div class="value">{patient_data.get('blood_pressure', 'N/A')}</div>
                    </div>
                    <div class="vital-sign">
                        <h3><i class="fas fa-temperature-high"></i> Temperature</h3>
                        <div class="value">{patient_data.get('temperature')}</div>
                    </div
                </div>
            </div>

            <div class="section">
                <h2><i class="fas fa-notes-medical"></i> Medical History</h2>
                <table>
                    <tr>
                        <th><i class="fas fa-smoking"></i> Smoking</th>
                        <td>{patient_data.get('smoking_habit', 'N/A')}</td>
                    </tr>
                    <tr>
                        <th><i class="fas fa-wine-glass-alt"></i> Drinking</th>
                        <td>{patient_data.get('drinking_habit', 'N/A')}</td>
                    </tr>
                </table>
            </div>

            <div class="section">
                <h2><i class="fas fa-chart-line"></i> Analysis Results</h2>
                
                <div class="chart-container">
                    <canvas id="heartDiseaseChart"></canvas>
                </div>

                <div class="risk-indicator" style="background-color: {_get_risk_color(results_data.get('risk_assessment_score', 0))}">
                    <h3>Risk Assessment Score</h3>
                    <div style="font-size: 3rem; font-weight: 600;">
                        {results_data.get('risk_assessment_score', 'N/A')}
                    </div>
                    <p>{_get_risk_label(results_data.get('risk_assessment_score', 0))}</p>
                </div>
            </div>

            <div class="button-container">
                <a href="#" onclick="window.print()" class="download-btn">
                    <i class="fas fa-download"></i> Download Report
                </a>
            </div>
        </div>

        <script>
            // Heart Disease Analysis Chart
            const ctx = document.getElementById('heartDiseaseChart').getContext('2d');
            new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: [
                        'Normal',
                        'Adenocarcinoma',
                        'Large Cell Carcinoma',
                        'Squamous Cell Carcinoma'
                    ],
                    datasets: [{{
                        data: [
                            {results_data.get('heart_disease', {}).get('class_probabilities', {}).get('normal', 0)},
                            {results_data.get('heart_disease', {}).get('class_probabilities', {}).get('adenocarcinoma', 0)},
                            {results_data.get('heart_disease', {}).get('class_probabilities', {}).get('large.cell.carcinoma', 0)},
                            {results_data.get('heart_disease', {}).get('class_probabilities', {}).get('squamous.cell.carcinoma', 0)}
                        ],
                        backgroundColor: [
                            '#4CAF50',
                            '#FFC107',
                            '#F44336',
                            '#9C27B0'
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom'
                        }},
                        title: {{
                            display: true,
                            text: 'Heart Disease Analysis Results',
                            font: {{
                                size: 16
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    filename = f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, 'w') as f:
        f.write(html_content)
    
    # Open in default browser
    webbrowser.open('file://' + os.path.realpath(filename))

def _get_risk_color(risk_score):
    """Get color based on risk score"""
    try:
        risk_score = float(risk_score)
        if risk_score < 30:
            return "#4CAF50"  # Green
        elif risk_score < 60:
            return "#FFC107"  # Yellow
        else:
            return "#F44336"  # Red
    except (TypeError, ValueError):
        return "#9E9E9E"  # Gray for N/A

def _get_risk_label(risk_score):
    """Get risk label based on score"""
    try:
        risk_score = float(risk_score)
        if risk_score < 30:
            return "Low Risk"
        elif risk_score < 60:
            return "Moderate Risk"
        else:
            return "High Risk"
    except (TypeError, ValueError):
        return "Risk Unknown"

# Example usage
if __name__ == "__main__":
    # Load your JSON data
    with open('complete_record.json', 'r') as f:
        patient_data = json.load(f)
    with open('output.json', 'r') as f:
        results_data = json.load(f)
    
    generate_enhanced_medical_report(patient_data, results_data)