from flask import Flask, render_template_string
import json
import os

app = Flask(__name__)


# Read JSON data from the file
def get_json_data():
    # Adjust the file name/path if needed
    file_path = os.path.join(os.path.dirname(__file__), 'output.json')
    with open(file_path, 'r') as f:
        return json.load(f)


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Patient Data Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS CDN -->
  <link 
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" 
    rel="stylesheet">
  <style>
    body {
      background: #f8f9fa;
    }
    .card {
      margin-bottom: 1.5rem;
    }
    .card-header {
      background-color: #007bff;
      color: white;
      font-weight: bold;
    }
    .data-key {
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div class="container my-5">
    <h1 class="text-center mb-4">Patient Data Dashboard</h1>
    <div id="data-container"></div>
  </div>

  <!-- Bootstrap JS Bundle -->
  <script 
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js">
  </script>

  <!-- Embed JSON data as a JS object -->
  <script>
    const jsonData = {{ data | tojson }};

    // Create a card for a given title and content
    function createCard(title, contentHtml) {
      const card = document.createElement('div');
      card.className = 'card shadow';

      const cardHeader = document.createElement('div');
      cardHeader.className = 'card-header';
      cardHeader.innerText = title;

      const cardBody = document.createElement('div');
      cardBody.className = 'card-body';
      cardBody.innerHTML = contentHtml;

      card.appendChild(cardHeader);
      card.appendChild(cardBody);

      return card;
    }

    // Recursively render JSON data
    function renderData(key, data) {
      let html = '';
      if (typeof data === 'object' && data !== null) {
        html += '<ul class="list-group list-group-flush">';
        for (const k in data) {
          html += '<li class="list-group-item">';
          html += '<span class="data-key">' + k + ':</span> ';
          if (typeof data[k] === 'object' && data[k] !== null) {
            html += renderData(k, data[k]);
          } else {
            html += data[k];
          }
          html += '</li>';
        }
        html += '</ul>';
      } else {
        html += data;
      }
      return html;
    }

    // When the page loads, build the UI
    window.onload = function() {
      const container = document.getElementById('data-container');
      for (const section in jsonData) {
        const content = renderData(section, jsonData[section]);
        // Convert underscores to spaces and uppercase
        const title = section.replace(/_/g, ' ').toUpperCase();
        const card = createCard(title, content);
        container.appendChild(card);
      }
    };
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    data = get_json_data()
    return render_template_string(html_template, data=data)


if __name__ == "__main__":
    app.run(debug=True, port=5021)
