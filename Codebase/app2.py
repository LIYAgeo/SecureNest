from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess

app = Flask(__name__)

# Route to serve the index.html as the root page
@app.route('/get_alerts', methods=['GET'])
def get_alerts():
    try:
        with open('alerts.json', 'r') as file:
            alerts = json.load(file)
        return jsonify(alerts)
    except Exception as e:
        print(f"Error reading alerts.json: {str(e)}")
        return jsonify([]), 500
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle login logic and serve the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simulate login success - add authentication logic here if needed
        return redirect(url_for('dashboard'))
    # Render login page on GET request
    return render_template('login.html')

# Route to serve the dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

# Route to start live video (for demonstration, assuming it runs `int2.py`)
@app.route('/start_live_video', methods=['POST'])
def start_live_video():
    try:
        # Run the int2.py script using subprocess
        subprocess.Popen(["python", "dummy1.py"])  # Make sure "python" is correct for your environment (e.g., use "python3" if necessary)
        return jsonify({"message": "Live video started!"})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
  # Set debug=False in a production environment







