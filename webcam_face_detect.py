import cv2
import face_recognition
from videocapturebufferless import VideoCaptureBufferless
import time
import keras 
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

if __name__ == '__main__':
    #emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    #emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}
    emotion_dict = ["Angry", "Disgust", "Scared", "Happy", "Sad", "Surprised", "Neutral"]
    model = load_model("emotion_detector_models/_mini_XCEPTION.106-0.65.hdf5", compile=False)
    
    cap = VideoCaptureBufferless("http://localhost:8081")
    cap.read()

    while True:
        rgb_frame = cap.read()

        #configure the ratio to get the desire performation. Trading detection rate
        #as it will have a hard time detecting small faces
        ratio = 0.5
        small_image=cv2.resize(rgb_frame, (0,0), fx=ratio, fy=ratio)


        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(small_image, model='cnn')
        #, model='cnn')
        for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            # Draw a label with a name below the face
            cv2.rectangle(small_image, (left, top), (right, bottom), (0, 255, 0))
            face_image = small_image[top:bottom, left:right]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image = cv2.resize(face_image, (48,48))
            #face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
            face_image = face_image.astype("float")/255.0
            face_image = img_to_array(face_image)
            face_image = np.expand_dims(face_image, axis=0)
            model_result = model.predict(face_image)

            #print(model_result[0])
            #print("Max is "+str(np.argmax(model_result)))
            predicted_class = np.argmax(model_result)
            
            #label_map = dict((v,k) for k,v in emotion_dict.items())
            predicted_label = emotion_dict[predicted_class]
            print("Probability is "+str(model_result[0][predicted_class]))

            cv2.putText(small_image,
                str(predicted_label),
                (left, top),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

        cv2.imshow('Video', small_image)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
