import cv2

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier('/usr/local/lib/python3.11/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

# Load the image of the specific face you want to recognize
target_face_path = cv2.imread('Michael.JPG')
if target_face_path.all() == None:
    raise Exception("could not Load Image!")
target_face = cv2.cvtColor(target_face_path, cv2.IMREAD_GRAYSCALE)

# Initialize the video capture object to access the webcam (0 indicates the default camera)
cap = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any faces are detected
    if len(faces) == 0:
        print("No faces detected")
    else:
        print(f"{len(faces)} face(s) detected")

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) containing the face
            face_roi = gray_frame[y:y+h, x:x+w]

            # Resize the ROI to match the size of the target face
            face_roi_resized = cv2.resize(face_roi, (target_face.shape[1], target_face.shape[0]))

            # Compare the resized ROI with the target face
            match_result = cv2.matchTemplate(face_roi_resized, target_face, cv2.TM_CCOEFF_NORMED)

            # Define a threshold for face matching
            threshold = 0.8

            # If the match result exceeds the threshold, unlock the lock
            if match_result >= threshold:
                print("Detected the target face - Unlocking the lock")
                # Code to unlock the lock goes here
            else:
                print("Detected a face but not the target face")

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with rectangles around the detected faces
    cv2.imshow('Face Recognition', frame)

    # Check for the 'q' key press or if the window is closed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
