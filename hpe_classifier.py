import os
import sys
import cv2

sys.path.append("../pose-tensorflow/")

from scipy.misc import imread

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

class HPEClassifier:
    def __init__(self, config_file):
        # Load and setup CNN part detector
        self.config = load_config(config_file)
        self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(self.config)



    # Expects an image to be of the form BGR (The preferred opencv way)
    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = self.sess.run(self.outputs, feed_dict={self.inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, self.config)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, self.config.stride)
        return Joints.from_tfpose(pose)


class Joints:


    def __init__(self):
        self.ankle_l = -1
        self.ankle_r = -1
        self.knee_l = -1
        self.knee_r = -1
        self.hip_l = -1
        self.hip_r = -1
        self.wrist_l = -1
        self.wrist_r = -1
        self.elbow_l = -1
        self.elbow_r = -1
        self.shoulder_l = -1
        self.shoulder_r = -1
        self.neck = -1
        self.forehead = -1
       
    def from_tfpose(values):
        joints = Joints()
        joints.ankle_l = (values[0][0], values[0][1])
        joints.knee_l =  (values[1][0], values[1][1])
        joints.hip_l = (values[2][0], values[2][1])
        joints.hip_r = (values[3][0], values[3][1])
        joints.knee_r = (values[4][0], values[4][1])
        joints.ankle_r = (values[5][0], values[5][1])
        joints.wrist_l = (values[6][0], values[6][1])
        joints.elbow_l = (values[7][0], values[7][1])
        joints.shoulder_l = (values[8][0], values[8][1])
        joints.shoulder_r = (values[9][0], values[9][1])
        joints.elbow_r = (values[10][0], values[10][1])
        joints.wrist_r = (values[11][0], values[11][1])
        joints.neck = (values[12][0], values[12][1])
        joints.forehead = (values[13][0], values[13][1])
        return joints

    def __str__(self):
        return str({'ankle_l': self.ankle_l,
         'knee_l': self.knee_l,
         'hip_l': self.hip_l,
         'hip_r': self.hip_r,
         'knee_r': self.knee_r,
         'ankle_r': self.ankle_r,
         'wrist_l': self.wrist_l,
         'elbow_l': self.elbow_l,
         'shoulder_l': self.shoulder_l,
         'shoulder_r': self.shoulder_r,
         'elbow_r': self.elbow_r,
         'wrist_r': self.wrist_r,
         'neck': self.neck,
         'forehead': self.forehead})

    
