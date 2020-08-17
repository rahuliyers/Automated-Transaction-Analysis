import numpy as np
from keras.models import model_from_yaml
from face_feature_extractor import FaceFeatureExtractor
from face_feature_preprocessor import FaceFeaturePreprocessor

#This will classify FAC units that are observed in the face
# The landmarks are https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
#It expects at least 4 consecutive image frames for a prediction

class FAUClassifier:

    #a dictionary to retrieve the correct index of the dependent for the desired code
    code_idx = {#Main AU
        1: 0, 2:1, 4:2, 5:3, 6:4, 7:5, 9:6, 10:7, 11:8,
        12:9, 14:10, 15:11, 16:12, 17:13, 18:14, 20:15, 
        21:16, 22:17, 23:18, 24:19, 25:20, 26:21, 27:22, 28:23,
        #Gross behavior AU
        29:24, 30:25, 31:26, 38:27, 39:28, 43:29, 45:30}


    def __init__(self, model_file, weights_file):
        with open(model_file, 'r') as f:
            loaded_model_yaml = f.read()
        
        self.model = model_from_yaml(loaded_model_yaml)
        self.model.load_weights(weights_file)
        self.extractor = FaceFeatureExtractor()
        self.preprocessor = FaceFeaturePreprocessor()

    def predict(self, images):
        feats = np.ndarray(shape=(len(images), 136))

        for i in range(len(images)):
            shape = self.extractor.extract(images[i])

            #skip processing - return no predictions
            if(shape is None):
                return (None,None) 

            for k in range(len(shape)):
                feats[i][(2*k)] = shape[k][0]
                feats[i][(2*k)+1] = shape[k][1]

        samples = np.array(self.preprocessor.process(feats))

        
        predictions = self.model.predict(samples.reshape((len(samples), 544,)))
        predictions = predictions > 0.5

        return (feats.reshape((4,68,2)), predictions)

