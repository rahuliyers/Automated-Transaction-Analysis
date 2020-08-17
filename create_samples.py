import os
import pickle
import itertools
import pandas as pd






xls_path = 'Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.xls'
FACS = pd.read_excel(xls_path, sheet_name="Sheet1",skiprows=3, names=[0,1,2],index=False)





with open('features.obj', 'rb') as ff:
    features = pickle.loads(ff.read())


samples = [] #the features of each session for each subject. Key is combination of subject/session
dependents = []


#sliding window across the feature sets (frames) with a width of 4 and a step of 1
#hence each sample will consist of 272 features


for i in range(FACS.shape[0]):
    r = FACS.ix[i]
    subjsess = os.path.join('S'+str(r[0]).zfill(3),str(r[1]).zfill(3))
    f = features[subjsess]
    for j in range(len(features[subjsess]) - 5):
        s = list(itertools.chain.from_iterable(f[j:(j+4)]))
        samples.append(s)
        dependents.append(r[2])

XY = {'X': samples, 'Y': dependents}

samples_file = 'samples.obj'


with open(samples_file, 'wb') as f:
    pickle.dump(XY, f)

