import cv2
import numpy as np
import pandas as pd

# Load the pre-trained face detection and recognition models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the attendance Excel sheet
attendance_file = 'attendance.xlsx'
attendance_df = pd.read_excel(attendance_file)

# Function to capture and recognize faces
def recognize_faces():
    cap = cv2.VideoCapture(0)  # Open the default camera

    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the face region
            face_img = gray[y:y + h, x:x + w]

            # Perform face recognition
            label, confidence = recognizer.predict(face_img)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if confidence < 100:  # Adjust this threshold based on your recognition model
                # Get the name from the attendance sheet based on the recognized label
                name = attendance_df.loc[attendance_df['ID'] == label, 'Name'].values[0]
                print('Recognized:', name)

                # Mark attendance in the Excel sheet
                attendance_df.loc[attendance_df['ID'] == label, 'Status'] = 'Present'

            else:
                name = 'Unknown'

            # Display the name on the frame
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Attendance System', frame)  # Display the frame

        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the updated attendance sheet
    attendance_df.to_excel(attendance_file, index=False)

# Train the face recognition model
def train_model():
    # Load the face images and labels for training
    faces = []  # List to store face images
    labels = []  # List to store corresponding labels

    # TODO: Add code to load face images and labels from a dataset

    # Train the recognizer
    recognizer.train(faces, np.array(labels))

# Main function
def main():
    train_model()  # Train the face recognition model
    recognize_faces()  # Start face recognition and attendance marking

if __name__ == '__main__':
    main()
