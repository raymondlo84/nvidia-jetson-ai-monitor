from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2

image1 = Image.open("face.jpg")
image_array1 = np.array(image1)
plt.imshow(image_array1)

image = face_recognition.load_image_file("face.jpg")
face_locations = face_recognition.face_locations(image)


