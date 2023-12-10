# -PYTHON-FaceRecognition


## Overview
This provides a implementation of face recognition using Python Language in Real-time detection. The project uses OPENCV and face_recognition library. The system loads known faces from specified directory, encodes them, and compares them with face captured through the webcam.

### Prerequisites
Make sure to install the required libraries before running the code:

**pip install face_recognition opencv-python numpy**

Getting Started

1.  Clone the repository to your local machine:

**git clone https://github.com/your-username/real-time-face-recognition.git
cd real-time-face-recognition**

2. Create a directory named images and place images of known people inside it.

3. Run the script:
**python face_recognition_system.py**

### How it Works
The script loads images of known people from the images directory and encodes their facial features using the face_recognition library.

The webcam feed is captured using OpenCV (cv2), and the captured frames are resized for efficiency.

The face_recognition library is used to locate and encode faces in each frame.

The encoded face is compared with the known faces, and the closest match is determined based on face distances.

If a match is found, the person's name is displayed on the frame along with a bounding box around their face.

Press 'q' to exit the application.
