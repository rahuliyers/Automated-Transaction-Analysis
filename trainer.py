import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import re


#a dictionary to retrieve the correct index of the dependent for the desired code
code_idx = {#Main AU
            1: 0, 2:1, 4:2, 5:3, 6:4, 7:5, 9:6, 10:7, 11:8,
            12:9, 14:10, 15:11, 16:12, 17:13, 18:14, 20:15, 
            21:16, 22:17, 23:18, 24:19, 25:20, 26:21, 27:22, 28:23,
            #Gross behavior AU
            29:24, 30:25, 31:26, 38:27, 39:28, 43:29, 45:30
            }


with open('samples.obj', 'rb') as f:
  samples = pickle.loads(f.read())


X = np.array(samples['X'])
raw_Y = np.array(samples['Y'])

n = len(X)
if n != len(raw_Y):
    print("FATAL: mismatch between lenghts of X and Y")
    exit(-1)

print("Samples: ", n)

np.random.seed(111)
print("Preparing data")
testIdx = np.random.uniform(0,1,n) > 0.7 # 70/ 30 split for cross validation

trainIdx = ~testIdx

#1-hot encode the dependents

Y = np.zeros(shape=(n, len(code_idx)))

for i in range(n):
    U = raw_Y[i].split('+')
    for u in U:
        if re.search(r'^\D',u):
            continue 
        u = re.sub(r'\D', '',u)
        
        if int(u) in code_idx:
            Y[(i, code_idx[int(u)])] = 1


        
for i in range(Y.shape[1]):
    print(i+1, " >>>>> ", sum(Y[:,i]))

X_train = X[trainIdx,]

Y_train = Y[trainIdx,]


X_test = X[testIdx,]
Y_test = Y[testIdx,]


print("Training set: ", len(X_train))
print("Test set: ", len(X_test))

print("Constructing model")

model = Sequential([
    Dense(256, activation='relu', input_shape=(544,)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(code_idx), activation='sigmoid'),
])

print("Configuring model")

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.8, nesterov=True)

model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])


print("Training model")
model.fit(X_train, Y_train,
          
          batch_size=4, epochs=500,
          validation_data=(X_test, Y_test))

model_yaml = model.to_yaml()
with open("face-action-unit-classifier.nn.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

model.save_weights("face-action-unit-classifier.nn.h5")
