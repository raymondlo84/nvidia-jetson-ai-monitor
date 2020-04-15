import cv2
import face_recognition
from videocapturebufferless import VideoCaptureBufferless

cap = VideoCaptureBufferless("http://localhost:8081")

while True:
    rgb_frame = cap.read()
    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame, model='cnn')
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # Draw a label with a name below the face
        cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0))
   
    cv2.imshow('Video', rgb_frame)
    if cv2.waitKey(1) == 27:
        exit(0)

cap.release()
cv2.destroyAllWindows()
