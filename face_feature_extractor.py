from imutils import face_utils
import pickle
import numpy as np
import dlib
import cv2
import pandas
import os
 

class FaceFeatureExtractor:
    def __init__(self, shape_predictor_file="shape_predictor_68_face_landmarks.dat"):
        p = os.path.join(os.getcwd(), shape_predictor_file)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(p)

    def extract(self, image):
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            return None

        # Make the prediction and transfom it to numpy array
        shape = self.predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)


        #Make nose tip the center of our plane (as it is the most central feature of the face) this is done to normalize the position of the features between samples
        #subtract feature 31 from all features
        center = (shape[30][0], shape[30][1])
        for k in range(len(shape)):
            shape[k][0] = shape[k][0] - center[0]
            shape[k][1] = shape[k][1] - center[1]

        return shape
