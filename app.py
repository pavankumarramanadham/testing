import cv2
import os
import numpy as np
import joblib
from flask import Flask, request, render_template, Response, jsonify
from datetime import date, datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sklearn.neighbors import KNeighborsClassifier
import base64
from io import BytesIO
from PIL import Image
import atexit  # Ensure proper cleanup on shutdown

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

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load Haarcascade face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ensure necessary directories exist
os.makedirs('static/faces', exist_ok=True)

# Load trained face recognition model
def load_model():
    model_path = 'static/face_recognition_model.pkl'
    return joblib.load(model_path) if os.path.exists(model_path) else None

model = load_model()

# Extract faces from an image (Convert to grayscale)
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return [cv2.resize(gray[y:y+h, x:x+w], (50, 50)) for (x, y, w, h) in faces] if len(faces) > 0 else []

# Identify a face using KNN model
def identify_face(face):
    global model
    if model:
        distances, indices = model.kneighbors(face.reshape(1, -1), n_neighbors=1)  # Get distance from nearest neighbor
        predicted_label = str(model.predict(face.reshape(1, -1))[0])  # Get the predicted label

        min_distance = distances[0][0]  # Distance to nearest neighbor
        confidence_threshold = 10000  # Adjust this value as needed (lower = stricter)

        # Only accept if prediction is confident & label exists
        if min_distance < confidence_threshold and predicted_label in os.listdir('static/faces'):
            return predicted_label  

    return None  # Face is unrecognized


# Train KNN model with stored face data
def train_model():
    face_data, labels = [], []
    registered_users = os.listdir('static/faces')

    for user in registered_users:
        user_folder = f'static/faces/{user}'
        for img_file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))

            face_data.append(img.flatten())  # Convert to 1D array
            labels.append(user)  # Use full "username_id" as label

    if face_data:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(face_data, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')  # Save model

        global model
        model = knn  # Reload new model

# Flask route to process new user registration
@app.route('/process_new_user', methods=['POST'])
def process_new_user():
    data = request.get_json()
    username, userid, images = data['username'], data['userid'], data['images']
    user_folder = f'static/faces/{username}_{userid}'
    os.makedirs(user_folder, exist_ok=True)
    
    for i, img_data in enumerate(images):
        with Image.open(BytesIO(base64.b64decode(img_data.split(',')[1]))) as img:
            img = img.resize((320, 240))
            img.save(f"{user_folder}/{username}_{i}.jpg")

    train_model()  # Train model after adding new user
    return jsonify({'message': f'User {username} added successfully!'})

# Video streaming setup
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

def generate_frames():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('home.html', totalreg=len(os.listdir('static/faces')), datetoday2=datetoday2)


# Process attendance with face recognition
@app.route('/process_attendance', methods=['POST'])
def process_attendance():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    with Image.open(BytesIO(base64.b64decode(image_data))) as img:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    faces = extract_faces(frame)
    if faces:
        identified_person = identify_face(faces[0])
        if identified_person:
            add_attendance(identified_person)
            return jsonify({'message': f'Attendance marked for {identified_person}'})
        else:
            # If no recognized face, return an alert message
            return jsonify({'message': 'Face not recognized. Please register first.'})

    return jsonify({'message': 'No face detected, try again'})


# Add attendance (prevents duplicate entries within 10 minutes)
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    
    last_entry = Attendance.query.filter_by(roll=userid, date=datetoday).order_by(Attendance.id.desc()).first()
    if last_entry:
        last_time = datetime.strptime(last_entry.time, "%H:%M:%S")
        if (datetime.now() - last_time).total_seconds() < 600:  # 10-minute buffer
            return

    db.session.add(Attendance(name=username, roll=userid, time=current_time, date=datetoday))
    db.session.commit()

# Shutdown and release camera
@app.route('/shutdown')
def shutdown():
    release_camera()
    return "Camera released"

def release_camera():
    if cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()

atexit.register(release_camera)  # Ensure camera release on exit

if __name__ == '__main__':
    app.run(debug=True)
