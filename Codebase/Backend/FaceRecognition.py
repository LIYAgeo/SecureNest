import cv2
import face_recognition
import os

# Path to the dataset
dataset_path = 'C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/Dataset B'

# Load images and encode faces
known_faces = []
known_names = []

# Loop through each person in the dataset folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)  # Get path to each person's folder
    if os.path.isdir(person_folder):  # Check if it's a directory
        for image_name in os.listdir(person_folder):  # Loop through images in the folder
            image_path = os.path.join(person_folder, image_name)  # Full path to the image
            image = face_recognition.load_image_file(image_path)  # Load the image
            encoding = face_recognition.face_encodings(image)  # Get the face encoding
            if encoding:  # Only add if encoding is found
                known_faces.append(encoding[0])  # Add encoding to list
                known_names.append(person_name)  # Add name to list
            else:
                print(f"No encoding found for {image_name}")

# Start video capture
video_path = 'C:/Users/asus/OneDrive/Desktop/Mini Project/Test Subjects/TEST 2.mp4'  # Replace with your actual video file path
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()  # Capture frame from video
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

    # Find faces in the resized frame
    face_locations = face_recognition.face_locations(rgb_small_frame)

    if face_locations:
        print(f"Detected {len(face_locations)} face(s).")  # Print the number of detected faces

        # Get face encodings
        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"

                # Use the first match if found
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Scale back face locations to original frame size
                top, right, bottom, left = [i * 4 for i in face_location]

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except Exception as e:
            print(f"Error processing frame: {e}")

    else:
        print("No faces detected.")

    # Display the frame with rectangles
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()



