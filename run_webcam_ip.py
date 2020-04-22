import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from videocapturebufferless import VideoCaptureBufferless

import face_recognition

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

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

    privacy = False

    while True:
        image = cam.read()

        #ratio = 0.5
        #small_image=cv2.resize(image, (0,0), fx=ratio, fy=ratio) 

        #process face
        #face_locations = face_recognition.face_locations(small_image, model='cnn')

       
        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        if privacy:
            image = blank_image.copy()
        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        #logger.debug('postprocess face+')
        #for face_location in face_locations:
            # Print the location of each face in this image
            #top, right, bottom, left = face_location
            # Draw a label with a name below the face
            #cv2.rectangle(image, (int(left/ratio), int(top/ratio)), (int(right/ratio), int(bottom/ratio)), (0, 255, 0))

 
        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow('TF Pose & Face Demo', image)
        fps_time = time.time()
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 80: #p key
            privacy = not privacy
        
        #logger.debug('finished+')
    cam.release()
    cv2.destroyAllWindows()
