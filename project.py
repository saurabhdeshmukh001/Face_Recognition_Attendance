import cv2
import face_recognition
import os
import numpy as np
import datetime

# Load images and encode known faces
known_faces_dir = "known_faces"
known_faces = []
known_names = []

for filename in os.listdir(known_faces_dir):
    image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(os.path.splitext(filename)[0])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for face_encoding in face_encodings:
        # Compare face encoding with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Find the best match index
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        # If there is a match, assign the name
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Draw a box and label around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
