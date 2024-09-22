import cv2
import face_recognition
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, dataset_path):
        self.known_faces = []
        self.known_names = []
        self.load_faces(dataset_path)

    def load_faces(self, dataset_path):
        # Load images and encode faces
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_folder):
                for image_name in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, image_name)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_faces.append(encodings[0])
                        self.known_names.append(person_name)

    def recognize_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Convert BGR to RGB

        # Find faces in the resized frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_names = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_faces, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]
            recognized_names.append((name, face_location))

        return recognized_names

def main():
    dataset_path = 'C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/Dataset B'
    video_path = 'C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/TEST 2.mp4'

    face_recognizer = FaceRecognizer(dataset_path)
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        recognized_faces = face_recognizer.recognize_faces(frame)

        for name, face_location in recognized_faces:
            top, right, bottom, left = [i * 4 for i in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


