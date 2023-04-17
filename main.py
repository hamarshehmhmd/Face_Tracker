import cv2
import numpy as np

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the cascade for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video capture object
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Failed to read frame from video capture object")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detect eyes in the face region of interest
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop through the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            # Write "Eyes Detected" above the eyes
            cv2.putText(frame, 'Eyes Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write "Human" above the face
        cv2.putText(frame, 'Human', (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face and Eye Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
