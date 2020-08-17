from fau_classifier import FAUClassifier
from hpe_classifier import HPEClassifier
import math
import numpy as np
import cv2
from scipy.spatial import distance


code_idx = FAUClassifier.code_idx

def near(a,b, d=5):
    #TODO: Should be < a value that is proportionally  near
    return distance.euclidean(a,b) < d


# A set of rules composed together to form an egostate classifier based on chapter 5 "I'm Ok, You're Ok"
# This classifier makes use of underlying machine learning algorithms to suffice the logical predicates of the rules
# This classifier expects that a person is facing toward the camera

class EgoStateClassifier:
    def __init__(self, fau_classifier, hpe_classifier, language_classifier):
        self.fau_classifier = fau_classifier
        self.hpe_classifier = hpe_classifier
        self.language_classifier = language_classifier

    #Process a stream of consecutive images (a video stream)
    def predict(self, images, sentences):

        indicator_track = []

        #VERBAL / TEXT

        
        sentence_annotations, sentence_predictions = self.language_classifier.predict(sentences['text'])

        for i in range(len(sentences['id'])):

            indicator_id = sentences['id'][i]
            indicators = EgoStateIndicators(indicator_id, 'Text')
            sp = sentence_predictions[i]
            sa = sentence_annotations[i]
            #
            # ADULT
            #

            #The sentence is interogative (who what when where why how ?)
            if sp.argmax() == 1:
                indicators.adult['interogative_sentence'] = True



            #
            # PARENT
            #

            #The sentence is commanding
            if sp.argmax() == 3:
                indicators.parent['authorative_sentence'] = True



            #
            # CHILD
            #

            #The sentence contains a comparative adjective/adverb
            for a in sa:
                if 'RBR' == a[1] or 'JJR' == a[1]:
                    indicators.child['comparative_sentence'] = True
                if 'JJS' == a[1] or 'RBS' == a[1]:
                    indicators.child['superlative_sentence'] = True

            indicator_track.append(indicators)
                    


        #VISUAL


        #slide across the frames with step 1 and size 4
        n = len(images)-4
        print("Batches: ",n)
        for i in range(n):
            print("Frame batch: ", i+1 , "of", n, '\r' )
            indicator_id = i+1
            indicators = EgoStateIndicators(indicator_id, 'Video')
            batch = images[i:(i+4)]

            face_points, facs = self.fau_classifier.predict(batch)

            #skip non processable frames
            if face_points is None or facs is None:
                continue

            facs = facs[0]


            last_image = cv2.cvtColor(batch[3], cv2.COLOR_BGR2RGB)
            hpe = self.hpe_classifier.predict(last_image) #use the last image of the batch of 4 frames, assuming the final position of the 4 frames as being the position

            #
            # ADULT
            #

            # tilted - nose_top(28) and chin(9) have too much angle
            nose_top = face_points[3][27]
            chin = face_points[3][8]
            #print("chin:", chin, "nose:",nose_top)
            nose_top_chin_angle = math.atan2(nose_top[1] - chin[1], nose_top[0] - chin[0])
            nose_top_chin_angle = (math.degrees(nose_top_chin_angle)+90)%90
            #print(nose_top_chin_angle)

            if abs(nose_top_chin_angle) > 5:
                indicators.adult['head_tilted'] = True
            

            

            # constant moving (between heard statements/transactions - a conjunction should be seen as 2 statements) - requires face state tracking


            #
            # CHILD
            #

            #quivering lips  - (FAC 8, then 25 continuously) - Requires state tracking

            #rolling eyes - Not possible yet (Need eye ball movement classification)

            #downcast eyes - not possible yet (Need eye ball movement classification)

            #laughter - Not thought on, requires planning


            #shrugging shoulders - Need common shoulder position, then when shoulders are raised above this position, shrugging of shoulders is present
            # shoulders above manubrium(start of neck above chest)
            # gradient between right shoulder and manubrium is negative and gradient between left shoulder and manubrium is positive

            clavical_r_grad = math.atan2(hpe.shoulder_r[1] - hpe.neck[1], hpe.shoulder_r[0] - hpe.neck[0])
            clavical_l_grad = math.atan2( hpe.neck[1]- hpe.shoulder_l[1],  hpe.neck[0] - hpe.shoulder_l[0])
            if  clavical_r_grad < 0 and  clavical_l_grad > 0:
                clavical_r_angle = (180-math.degrees(clavical_r_grad))%180 
                clavical_l_angle = (180-math.degrees(clavical_l_grad))%180
                #print("Clav R true angle:", math.degrees(clavical_r_grad))
                #print("Clav R:", clavical_r_angle)
                #print("Clav L true angle:", math.degrees(clavical_l_grad))
                #print("Clav L:", clavical_l_angle )
                if clavical_r_angle > 10 and clavical_l_angle > 10:
                    indicators.child['shrug_shoulders'] = True



            #hand raising (permission) - when the wrist is inline with or above the neck and further across from shoulder (and not both hands up)
            #Left hand check and then 
            #Right hand check
            if (hpe.wrist_l[1] >= hpe.neck[1] and hpe.wrist_r[1] < hpe.neck[1] and  hpe.wrist_l[0] > hpe.shoulder_l[0]) or (hpe.wrist_r[1] >= hpe.neck[1] and hpe.wrist_l[1] < hpe.neck[1] and hpe.wrist_r[0] < hpe.shoulder_r[0]): 
                indicators.child['permission_hand'] = True

            #nose thumbing - when the wrist is centered with the chin (left/right)
            #We use the facial points of the last frame
            if near(hpe.wrist_l, face_points[3][8]) or near(hpe.wrist_r, face_points[3][8]):
                indicators.child['nose_thumbing'] = True
                

            #
            # PARENT
            #


            #Furrowed Brow - FAC 4 (brows lowered)
            if facs[code_idx[4]]:
                indicators.parent['furrowed_brow'] = True

            #Pursed Lips - FAC 18
            if facs[code_idx[18]]:
                indicators.parent['pursed_lips'] = True

            #horrified look - Surprise ( FAC 1 + 2 + 5 + 26 )
            if facs[code_idx[1]] and facs[code_idx[2]] and facs[code_idx[5]] and facs[code_idx[26]]:
                indicators.parent['horrified_look'] = True


            # arms folded across chest (wrists near opposite elbows and below the neck)
            # only check 1 elbow's height, if the wrists are near the elbows, the elbows are generally the same height
            if near(hpe.wrist_l, hpe.elbow_r) and near(hpe.wrist_r, hpe.wrist_l) and hpe.elbow_r[1] < hpe.neck[1]: 
                indicators.parent['arms_across_chest'] = True
            
            # Hands on hips
            if near(hpe.hip_l, hpe.wrist_l) and near(hpe.hip_r, hpe.wrist_l):
                indicators.parent['hands_on_hips'] = True

            # wringing fingers  (for now, if wrists are close together)
            if near(hpe.wrist_l, hpe.wrist_r):
                indicators.parent['wringing_fingers'] = True

            indicator_track.append(indicators)

        return indicator_track


# Represents the events emitted by the ego state classifier
# synonymous to the annotations emitted by the ego state classifier
class EgoStateIndicators(object):
    def __init__(self, idx, channel=None):
        self.idx = idx
        self.channel = channel
        self.adult = {}
        self.child = {}
        self.parent = {}
        
    def count(self):
        return len(self.adult) + len(self.child) + len(self.parent)

    def __str__(self):
        return "ID = " + str(self.idx) + "; Adult = " + str(self.adult) + "; Child = " + str(self.child) + "; Parent = " + str(self.parent)

