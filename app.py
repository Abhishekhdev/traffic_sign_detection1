from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import numpy as np
import pickle  # For loading the SVM model
from werkzeug.utils import secure_filename
import pyttsx3  # Offline Text-to-Speech
import threading  # For running TTS asynchronously
#from pymongo import MongoClient
from flask import Flask, render_template, redirect, url_for, request, session, flash, Response
from pymongo import MongoClient
import bcrypt
import cv2
import numpy as np
#from gtts import gTTS
#from playsound import playsound
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"


# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client['traffic_detection']
users_collection = db['users']


# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained SVM model and class labels
with open("models/traffic_sign_svm_model11.pkl", "rb") as f:
    model, class_labels = pickle.load(f)

# Reverse the class labels dictionary to get names from indices
class_names = {idx: label for label, idx in class_labels.items()}

# Function to preprocess the image for SVM
def preprocess_image(img_path):
    """Preprocess the image to match the SVM model input."""
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (100, 100))  # Resize to match training dimensions
    img_flattened = img_resized.flatten()  # Flatten the image to a 1D vector
    return img_flattened.reshape(1, -1)  # Return as a 2D array with one sample

# Function to detect the traffic sign
def detect_traffic_sign(img_path):
    """Predict the traffic sign using the trained SVM model."""
    img_preprocessed = preprocess_image(img_path)
    probabilities = model.predict_proba(img_preprocessed)[0] # Predict class
    max_prob = max(probabilities)
    threshold = 0.80  # Set a confidence threshold (80%)


    if max_prob >= threshold:
        predicted_class = np.argmax(probabilities)
        return class_names[predicted_class]
    else:
        return "Unknown sign"
    # return class_names[prediction]

# Function to generate and play voice output using pyttsx3 in a separate thread
def generate_voice_output(text):
    """Generate and play voice output asynchronously."""
    def speak():
        tts_engine = pyttsx3.init()  # Initialize the engine inside the thread
        tts_engine.say(text)
        tts_engine.runAndWait()
    
    # Run the TTS in a separate thread to avoid blocking the main Flask thread
    tts_thread = threading.Thread(target=speak)
    tts_thread.start()

# Route for the home page
@app.route('/')
def index():
    """Render the home page."""
    return render_template('landing.html')

# Route to handle image upload and traffic sign detection
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle image upload and traffic sign detection."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Detect traffic sign
            detected_sign = detect_traffic_sign(filepath)

            # Generate voice output for the detected sign
            speech_text = f"The detected traffic sign is {detected_sign}"
            generate_voice_output(speech_text)

            return render_template('upload.html', uploaded_image=filepath, detected_sign=detected_sign)

    return render_template('upload.html', uploaded_image=None, detected_sign=None)

# ------------------ VIDEO FEED FUNCTIONALITY ------------------

# Initialize the video capture (use 0 for default webcam)
video_capture = cv2.VideoCapture(0)

def generate_frames():
    """Video streaming generator function."""
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route to serve video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------ END OF VIDEO FEED FUNCTIONALITY ------------------

# Registration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Check if the user already exists
        if users_collection.find_one({'username': username}):
            flash("Username already exists!", "error")
            return redirect(url_for('register'))

        # Insert new user into the database
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))

        flash("Invalid username or password!", "error")
        return redirect(url_for('login'))

    return render_template('login.html')

# Dashboard (Traffic Detection)
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash("Please log in first!", "error")
        return redirect(url_for('login'))

    return render_template('index.html', username=session['username'])

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully!", "success")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
