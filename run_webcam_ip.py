import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from videocapturebufferless import VideoCaptureBufferless

#for face recognition and emotion analysis
import face_recognition
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
happy = 0

def find_face_emotion(image, image_draw):
    global happy
    emotion_dict = ["Angry", "Disgust", "Scared", "Happy", "Sad", "Surprised", "Neutral"]
    ratio = 0.5
    small_image=cv2.resize(image, (0,0), fx=ratio, fy=ratio) 

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
        face_image = face_image.astype("float")/255.0
        face_image = img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        model_result = model.predict(face_image)

        predicted_class = np.argmax(model_result)

        predicted_label = emotion_dict[predicted_class]
        #print("Emotion Probability of "+predicted_label+" is "+str(model_result[0][predicted_class]))
        if predicted_class == 3:
            happy = happy + 1
        cv2.putText(image_draw,
                    str(predicted_label),
                    (int(left/0.5), int(top/0.5)),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.rectangle(image_draw, (int(left/ratio), int(top/ratio)), (int(right/ratio), int(bottom/ratio)), (0, 255, 0))
        cv2.putText(image_draw,
                    "Happy Counter: "+str(happy),
                    (image_draw.shape[1]-170, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = VideoCaptureBufferless("http://localhost:8081")

    image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    image_draw =  np.zeros(image.shape, dtype=np.uint8)
    privacy = False
    model = load_model("emotion_detector_models/_mini_XCEPTION.106-0.65.hdf5", compile=False)

    while True:
        #get a frame from the IP camera
        image = cam.read()
        

        #image to draw on screen
        if privacy:
            image_draw = blank_image.copy()
        else:
            image_draw = image.copy()

        #perform the face detection and emotion analysis
        find_face_emotion(image, image_draw)

        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        #logger.debug('postprocess+')
        image_draw = TfPoseEstimator.draw_humans(image_draw, humans, imgcopy=False)
        
        
        #logger.debug('show+')
        cv2.putText(image_draw,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow('TF Pose & Face Demo', image_draw)
        fps_time = time.time()

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 80: #p key
            privacy = not privacy
        
        #logger.debug('finished+')
    cam.release()
    cv2.destroyAllWindows()
