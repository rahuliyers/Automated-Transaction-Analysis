# This project was created using Cohn-Kanada dataset
# http://www.pitt.edu/~jeffcohn/biblio/Cohn-Kanade_Database.pdf
# Sign their licence agreement and obtain access to the dataset in order to perform feature extraction.

from imutils import face_utils
import pickle
import numpy as np
import dlib
import cv2
import pandas
import os
from face_feature_extractor import FaceFeatureExtractor
 


root_path = os.getcwd()
data_path = os.path.join(root_path, 'cohn-kanade')



xls_path = 'Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.xls'
FACS = pandas.read_excel(xls_path, sheet_name="Sheet1",skiprows=3, names=[0,1,2],index=False)


features = {} #the features of each session for each subject. Key is combination of subject/session

extractor = FaceFeatureExtractor()


#extract features for the dataset 
for i in range(FACS.shape[0]):
    print("NOW PROCESSING ", i)
    # <data_path>/<Subject>/<Session>
    r = FACS.ix[i]
    subjsess = os.path.join('S'+str(r[0]).zfill(3),str(r[1]).zfill(3))
    d = os.path.join(data_path, subjsess)

    files = os.listdir(d)
    feats = np.ndarray(shape=(len(files), 136))



    for fi in range(len(files)):
        image = cv2.imread(os.path.join(d,files[fi]), cv2.IMREAD_COLOR)
        
        shape = extractor.extract(image)
        if shape is None:
            continue

        for k in range(len(shape)):
            feats[fi][(2*k)] = shape[k][0]
            feats[fi][(2*k)+1] = shape[k][1]

    features[subjsess] = feats

outfile = open('features.obj','wb')            

pickle.dump(features, outfile)
