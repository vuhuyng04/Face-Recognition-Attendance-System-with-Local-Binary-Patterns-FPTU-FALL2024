#================================== Importing Libraries ==========================
from flask import Flask, render_template, request, redirect, url_for, flash, Response
import cv2
import numpy as np
from PIL import Image
import pickle
import os
import datetime
import time
import pandas as pd
#============================= Flask app initialization ===========================
app = Flask(__name__)
app.secret_key = 'secretkey'


# Global variables
attendance_status = {}

# Helper functions
def save_face_data(faces, labels):
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

    with open(faces_file_path, 'wb') as f:
        pickle.dump(faces, f)
    with open(labels_file_path, 'wb') as f:
        pickle.dump(labels, f)

#=================== Load data from the saved files ========================
def load_data():
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

    faces = []
    labels = []

    try:
        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                faces = pickle.load(f)
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                labels = pickle.load(f)
    except (IOError, pickle.UnpicklingError) as e:
        flash(f"Failed to load data: {e}", 'error')

    return faces, labels





#=================== Draw bounding box around detected faces ========================
def draw_ai_tech_bbox(frame, x, y, w, h, name, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

class FaceDetectionTransformer:
    def __init__(self, recognizer, label_to_name):
        self.recognizer = recognizer
        self.label_to_name = label_to_name
        self.detected_names = {}

    def transform(self, frame):
        global attendance_status

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_locations:
            face = gray_frame[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(face)
            name = self.label_to_name.get(label, "Unknown")

            draw_ai_tech_bbox(frame, x, y, w, h, name)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name not in self.detected_names:
                self.detected_names[name] = timestamp
                attendance_status[name] = True
            else:
                last_saved_timestamp = datetime.datetime.strptime(self.detected_names[name], "%Y-%m-%d %H:%M:%S")
                current_timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                if (current_timestamp - last_saved_timestamp).seconds > 60:
                    self.detected_names[name] = timestamp
                    attendance_status[name] = True

        return frame
    

#=================== Collect data from camera ========================
def collect_data_from_camera(name, duration=10):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    labels = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_locations:
            face = gray_frame[y:y + h, x:x + w]
            faces.append(face)
            labels.append(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Collecting Face Data', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                existing_faces = pickle.load(f)
            faces = existing_faces + faces
        else:
            faces = faces

        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                existing_labels = pickle.load(f)
            labels = existing_labels + labels
        else:
            labels = labels

        save_face_data(faces, labels)
        flash("Data collected successfully!", 'success')
    else:
        flash("No data collected.", 'warning')


#=================== Load data from images ========================
def load_data_from_images(images, name):
    faces = []
    labels = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for img in images:
        img = np.array(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_locations:
            face = gray_img[y:y + h, x:x + w]
            faces.append(face)
            labels.append(name)

    if len(faces) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                existing_faces = pickle.load(f)
            faces = existing_faces + faces
        else:
            faces = faces

        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                existing_labels = pickle.load(f)
            labels = existing_labels + labels
        else:
            labels = labels

        save_face_data(faces, labels)
        flash("Images uploaded and data saved successfully!", 'success')
    else:
        flash("No data collected.", 'warning')


#=================== Routes ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect_data', methods=['GET', 'POST'])
def collect_data():
    if request.method == 'POST':
        name = request.form['name']
        method = request.form['method']
        if method == 'Upload Image':
            uploaded_files = request.files.getlist('files')
            images = [Image.open(file) for file in uploaded_files]
            load_data_from_images(images, name)
        elif method == 'Use Camera':
            duration = int(request.form['duration'])
            global collect_name, collect_duration
            collect_name = name
            collect_duration = duration
            return redirect(url_for('collect_camera_data'))
        return redirect(url_for('index'))
    return render_template('collect_data.html')

@app.route('/collect_camera_data')
def collect_camera_data():
    return render_template('collect_camera_data.html')

@app.route('/video_feed_collect')
def video_feed_collect():
    return Response(gen_frames_collect(), mimetype='multipart/x-mixed-replace; boundary=frame')


#=================== Generate frames for collecting data ========================
def gen_frames_collect():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    labels = []
    start_time = time.time()

    while time.time() - start_time < collect_duration:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_locations:
                face = gray_frame[y:y + h, x:x + w]
                faces.append(face)
                labels.append(collect_name)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                existing_faces = pickle.load(f)
            faces = existing_faces + faces
        else:
            faces = faces

        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                existing_labels = pickle.load(f)
            labels = existing_labels + labels
        else:
            labels = labels

        save_face_data(faces, labels)
        flash("Data collected successfully!", 'success')
    else:
        flash("No data collected.", 'warning')


#=================== Face Recognition ========================
@app.route('/face_recognition')
def face_recognition():
    faces, labels = load_data()

    if len(faces) == 0 or len(labels) == 0:
        flash("No data available for training!", 'error')
        return redirect(url_for('index'))

    # Add a dummy "Unknown" class if there is only one class
    if len(set(labels)) < 2:
        faces.append(np.zeros_like(faces[0]))
        labels.append("Unknown")

    # Convert labels to integers
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([np.array(face, dtype=np.uint8) for face in faces], np.array(int_labels, dtype=np.int32))

    # Create a reverse mapping from integer labels to names
    int_to_label = {v: k for k, v in label_to_int.items()}

    global face_detection_transformer
    face_detection_transformer = FaceDetectionTransformer(recognizer, int_to_label)

    return render_template('face_recognition.html')

#=================== Video feed for face recognition ========================
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = face_detection_transformer.transform(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


#================================== Importing Libraries ==========================
from flask import Flask, render_template, request, redirect, url_for, flash, Response
import cv2
import numpy as np
from PIL import Image
import pickle
import os
import datetime
import time
import pandas as pd
#============================= Flask app initialization ===========================
app = Flask(__name__)
app.secret_key = 'secretkey'


# Global variables
attendance_status = {}

# Helper functions
def save_face_data(faces, labels):
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

    with open(faces_file_path, 'wb') as f:
        pickle.dump(faces, f)
    with open(labels_file_path, 'wb') as f:
        pickle.dump(labels, f)

#=================== Load data from the saved files ========================
def load_data():
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

    faces = []
    labels = []

    try:
        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                faces = pickle.load(f)
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                labels = pickle.load(f)
    except (IOError, pickle.UnpicklingError) as e:
        flash(f"Failed to load data: {e}", 'error')

    return faces, labels





#=================== Draw bounding box around detected faces ========================
def draw_ai_tech_bbox(frame, x, y, w, h, name, color=(0, 255, 0), thickness=2):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

class FaceDetectionTransformer:
    def __init__(self, recognizer, label_to_name):
        self.recognizer = recognizer
        self.label_to_name = label_to_name
        self.detected_names = {}

    def transform(self, frame):
        global attendance_status

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_locations:
            face = gray_frame[y:y + h, x:x + w]
            label, confidence = self.recognizer.predict(face)
            name = self.label_to_name.get(label, "Unknown")

            draw_ai_tech_bbox(frame, x, y, w, h, name)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name not in self.detected_names:
                self.detected_names[name] = timestamp
                attendance_status[name] = True
            else:
                last_saved_timestamp = datetime.datetime.strptime(self.detected_names[name], "%Y-%m-%d %H:%M:%S")
                current_timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                if (current_timestamp - last_saved_timestamp).seconds > 60:
                    self.detected_names[name] = timestamp
                    attendance_status[name] = True

        return frame
    

#=================== Collect data from camera ========================
def collect_data_from_camera(name, duration=10):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    labels = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_locations:
            face = gray_frame[y:y + h, x:x + w]
            faces.append(face)
            labels.append(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Collecting Face Data', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                existing_faces = pickle.load(f)
            faces = existing_faces + faces
        else:
            faces = faces

        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                existing_labels = pickle.load(f)
            labels = existing_labels + labels
        else:
            labels = labels

        save_face_data(faces, labels)
        flash("Data collected successfully!", 'success')
    else:
        flash("No data collected.", 'warning')


#=================== Load data from images ========================
def load_data_from_images(images, name):
    faces = []
    labels = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for img in images:
        img = np.array(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in face_locations:
            face = gray_img[y:y + h, x:x + w]
            faces.append(face)
            labels.append(name)

    if len(faces) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                existing_faces = pickle.load(f)
            faces = existing_faces + faces
        else:
            faces = faces

        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                existing_labels = pickle.load(f)
            labels = existing_labels + labels
        else:
            labels = labels

        save_face_data(faces, labels)
        flash("Images uploaded and data saved successfully!", 'success')
    else:
        flash("No data collected.", 'warning')


#=================== Routes ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/collect_data', methods=['GET', 'POST'])
def collect_data():
    if request.method == 'POST':
        name = request.form['name']
        method = request.form['method']
        if method == 'Upload Image':
            uploaded_files = request.files.getlist('files')
            images = [Image.open(file) for file in uploaded_files]
            load_data_from_images(images, name)
        elif method == 'Use Camera':
            duration = int(request.form['duration'])
            global collect_name, collect_duration
            collect_name = name
            collect_duration = duration
            return redirect(url_for('collect_camera_data'))
        return redirect(url_for('index'))
    return render_template('collect_data.html')

@app.route('/collect_camera_data')
def collect_camera_data():
    return render_template('collect_camera_data.html')

@app.route('/video_feed_collect')
def video_feed_collect():
    return Response(gen_frames_collect(), mimetype='multipart/x-mixed-replace; boundary=frame')


#=================== Generate frames for collecting data ========================
def gen_frames_collect():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    labels = []
    start_time = time.time()

    while time.time() - start_time < collect_duration:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in face_locations:
                face = gray_frame[y:y + h, x:x + w]
                faces.append(face)
                labels.append(collect_name)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

    if len(faces) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        labels_file_path = os.path.join(os.getcwd(), 'labels.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                existing_faces = pickle.load(f)
            faces = existing_faces + faces
        else:
            faces = faces

        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'rb') as f:
                existing_labels = pickle.load(f)
            labels = existing_labels + labels
        else:
            labels = labels

        save_face_data(faces, labels)
        flash("Data collected successfully!", 'success')
    else:
        flash("No data collected.", 'warning')


#=================== Face Recognition ========================
@app.route('/face_recognition')
def face_recognition():
    faces, labels = load_data()

    if len(faces) == 0 or len(labels) == 0:
        flash("No data available for training!", 'error')
        return redirect(url_for('index'))

    # Add a dummy "Unknown" class if there is only one class
    if len(set(labels)) < 2:
        faces.append(np.zeros_like(faces[0]))
        labels.append("Unknown")

    # Convert labels to integers
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([np.array(face, dtype=np.uint8) for face in faces], np.array(int_labels, dtype=np.int32))

    # Create a reverse mapping from integer labels to names
    int_to_label = {v: k for k, v in label_to_int.items()}

    global face_detection_transformer
    face_detection_transformer = FaceDetectionTransformer(recognizer, int_to_label)

    return render_template('face_recognition.html')

#=================== Video feed for face recognition ========================
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = face_detection_transformer.transform(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


import csv

#=================== Attendance Log ========================
@app.route('/attendance_log', methods=['GET', 'POST'])
def attendance_log():
    global attendance_status

    faces, labels = load_data()
    if len(faces) == 0 or len(labels) == 0:
        flash("No data available for attendance log!", 'error')
        return redirect(url_for('index'))

    # Add a dummy "Unknown" class if there is only one class
    if len(set(labels)) < 2:
        faces.append(np.zeros_like(faces[0]))
        labels.append("Unknown")

    # Convert labels to integers
    label_to_int = {label: idx for idx, label in enumerate(set(labels))}
    int_labels = [label_to_int[label] for label in labels]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([np.array(face, dtype=np.uint8) for face in faces], np.array(int_labels, dtype=np.int32))

    # Create a reverse mapping from integer labels to names
    label_to_name = {v: k for k, v in label_to_int.items()}

    # Initialize attendance status if not already initialized
    if not attendance_status:
        attendance_status = {name: False for name in set(labels)}

    if request.method == 'POST':
        if 'reset' in request.form:
            attendance_status = {name: False for name in set(labels)}
            save_attendance_to_csv(attendance_status)
            flash("Attendance status reset successfully!", 'success')

    save_attendance_to_csv(attendance_status)
    return render_template('attendance_log.html', attendance_status=attendance_status)

def save_attendance_to_csv(attendance_status):
    with open('attendance.csv', 'w', newline='') as csvfile:
        fieldnames = ['Name', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for name, status in attendance_status.items():
            writer.writerow({'Name': name, 'Status': 'Present' if status else 'Absent'})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)