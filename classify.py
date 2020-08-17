from fau_classifier import FAUClassifier
from hpe_classifier import HPEClassifier
from ego_state_classifier import EgoStateClassifier
from language_classifier import LanguageClassifier
import numpy as np
import cv2

from scipy.misc import imread

from util.config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import nltk
import pickle
import pandas as pd
import os


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

cap = cv2.VideoCapture('demo/speaker2.mp4')

frames = []
while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        print("Broken: ", len(frames))
        break

    frames.append(frame)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

hpe_classifier = HPEClassifier('config/pose_cfg.yaml')
fau_classifier = FAUClassifier('face-action-unit-classifier.nn.yaml', 'face-action-unit-classifier.nn.h5')
language_classifier = LanguageClassifier()

ego_state_classifier = EgoStateClassifier(fau_classifier, hpe_classifier, language_classifier)



sentences = {'text':["How are you doing?", "Where will we go today?", "The world is fantastic!", "I will have a more competitive role", "Sometimes we see what we want to see.", "There are too many of them.", "Do your homework!", "How many times have I told you?"], 'id':[1,26,48,73,92,116,130, 140]}


indicators = ego_state_classifier.predict(frames, sentences)

print("Indicators: ", " <> ".join(str(x) for x in indicators))

adult_score = sum(len(i.adult) for i in indicators)
child_score = sum(len(i.child) for i in indicators)
parent_score = sum(len(i.parent) for i in indicators)
print("Adult score:", adult_score)
print("Child score:", child_score)
print("Parent score:", parent_score)

output_file = 'ego_state_indicators.obj'

with open(output_file, 'wb') as f:
    pickle.dump( indicators, f)

sentence_file = 'sentences.csv'
os.makedirs('static', exist_ok=True)

df = pd.DataFrame(sentences)
df.to_csv(os.path.join('static',sentence_file))

sentence_dict = {}
for i in range(len(sentences['id'])):
    sentence_dict[sentences['id'][i]] = sentences['text'][i]

with open('sentences.obj', 'wb') as f:
    pickle.dump(sentence_dict, f)

