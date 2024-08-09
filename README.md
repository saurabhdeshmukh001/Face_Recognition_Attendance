This project is a Face Recognition and Attendance System built using OpenCV and CSV files. The system captures images of individuals, recognizes their faces, and records their attendance in a CSV file. It is an efficient solution for automating the process of attendance tracking in educational institutions or workplaces.

Table of Contents Features Technologies Used Installation Usage Project Structure Contributing License Acknowledgements Features Face Detection: Detects faces in real-time using a webcam. Face Recognition: Recognizes and identifies registered faces. Attendance Tracking: Records attendance with the date and time in a CSV file. CSV Export: Easily export attendance records to a CSV file. Technologies Used Programming Language: Python Libraries: OpenCV (for face detection and recognition) NumPy (for numerical operations) Pandas (for CSV handling) Installation Clone the Repository:

bash Copy code git clone https://github.com/Saurabhdawkhar/Face-recognition-attendance.git cd face-recognition-attendance Install the Required Packages:

Make sure you have Python 3.x installed. Install the required libraries using pip:

bash Copy code pip install opencv-python pip install numpy pip install pandas Prepare the Dataset:

Collect images of individuals you want to recognize. Organize the images in folders where each folder is named after the person. Place the dataset in the dataset/ directory. Run the Training Script:

Run the script to train the model with the dataset:

bash Copy code python train_model.py Usage Run the Face Recognition System:

bash Copy code python face_recognition.py This will start the webcam, and the system will begin recognizing faces and recording attendance.

Check Attendance Records:

The attendance records are saved in attendance.csv. You can open this file to view the recorded attendance.

Project Structure bash Copy code face-recognition-attendance/ │ ├── dataset/ # Directory containing training images │ ├── face_recognition.py # Main script for running face recognition ├── train_model.py # Script for training the model ├── attendance.csv # CSV file to store attendance records ├── README.md # Project README file ├── requirements.txt # List of required packages └── .gitignore # Git ignore file Contributing Contributions are welcome! If you have any suggestions or improvements, feel free to fork the repository, make your changes, and submit a pull request.
