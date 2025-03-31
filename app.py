import cv2
import os
import numpy as np
import joblib
import logging
from flask import Flask, request, render_template, Response, jsonify
from datetime import date, datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier
import base64
from io import BytesIO
from PIL import Image

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask app initialization
app = Flask(__name__)

# SQLAlchemy setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define Attendance model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(100), nullable=False)

# Initialize database
with app.app_context():
    db.create_all()

# Define date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load Haarcascade face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load trained face recognition model
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

    
def load_model():
    try:
        if os.path.exists('static/face_recognition_model.pkl'):
            return joblib.load('static/face_recognition_model.pkl')
    except Exception as e:
        logging.error(f"Error loading model: {e}")
    return None

model = load_model()

# Extract faces from image (Convert to grayscale)
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return [cv2.resize(gray[y:y+h, x:x+w], (50, 50), interpolation=cv2.INTER_AREA) for (x, y, w, h) in faces] if len(faces) > 0 else []

def identify_face(face):
    global model
    if model:
        distances, indices = model.kneighbors(face.reshape(1, -1), n_neighbors=1)
        min_distance = distances[0][0]  # Distance to the closest known face
        
        threshold = 30  # Tune this value as needed
        if min_distance > threshold:
            return None  # Face is unknown
        
        return model.predict(face.reshape(1, -1))[0]  # Recognized face
    return None

# Train the face recognition model
def train_model():
    try:
        faces, labels = [], []
        for user in os.listdir('static/faces'):
            for imgname in os.listdir(f'static/faces/{user}'):
                img = cv2.imread(f'static/faces/{user}/{imgname}', cv2.IMREAD_GRAYSCALE)
                faces.append(cv2.resize(img, (50, 50)).ravel())
                labels.append(user)
        
        if faces:
            global model
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(np.array(faces), labels)
            joblib.dump(model, 'static/face_recognition_model.pkl')
    except Exception as e:
        logging.error(f"Error training model: {e}")

# Video streaming function
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        logging.info("Camera released after streaming")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('home.html', totalreg=len(os.listdir('static/faces')), datetoday2=datetoday2)

@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces = extract_faces(frame)
        if faces:
            identified_person = identify_face(faces[0])
            if identified_person:
                add_attendance(identified_person)
                return jsonify({'message': f'Attendance marked for {identified_person}'}), 200
            else:
                return jsonify({'message': 'Face not recognized. Please register first.'}), 400

        return jsonify({'message': 'No face detected, try again'}), 400
    except Exception as e:
        logging.error(f"Error processing attendance: {e}")
        return jsonify({'message': 'Error processing attendance'}), 500

# Add attendance
def add_attendance(name):
    try:
        username, userid = name.split('_')
        current_time = datetime.now().strftime("%H:%M:%S")
        last_entry = Attendance.query.filter_by(roll=userid, date=datetoday).order_by(Attendance.id.desc()).first()
        
        if last_entry:
            last_time = datetime.strptime(last_entry.time, "%H:%M:%S")
            if (datetime.now() - last_time).total_seconds() < 600:
                return  # Skip if the last entry was within 10 minutes

        db.session.add(Attendance(name=username, roll=userid, time=current_time, date=datetoday))
        db.session.commit()
    except Exception as e:
        logging.error(f"Error adding attendance: {e}")

@app.route('/shutdown')
def shutdown():
    logging.info("Server shutting down")
    return "Server shutting down"

if __name__ == '__main__':
    app.run(debug=True)
