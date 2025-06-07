import cv2
from keras.models import load_model
import numpy as np
# Load pre-trained FER model
emotion_model = load_model('facialemotionmodel.h5')
# Load Haar cascade for face detection 
#classifier use for 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start capturing video from the default camera (you can change the argument to use a different camera)
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for face from the grayscale frame
        roi_gray = gray[y:y + h, x:x + w]
        # Resize the ROI to match the input size of the FER model (48x48 pixels)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        # Normalize the pixel values to be in the range [0, 1]
        roi_gray = roi_gray / 255.0
        # Reshape the image to match the input shape expected by the model
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        # Make prediction using the FER model
        emotion_prediction = emotion_model.predict(roi_gray)
        # Get the predicted emotion label
        emotion_label = np.argmax(emotion_prediction)
        # Define the emotions corresponding to the label
        emotions = ["1 :Angry", "2 :Disgust", "3 :Fear", "4 :Happy", "5 :Neutral", "6 :Surprise", "7 :Sad"]
        # Display the emotion label on the frame
        cv2.putText(frame, emotions[emotion_label], (x -10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()