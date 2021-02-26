### Swtich to Python3.7.3 64bit Conda for this sample.
### For all other samples you can use Python 3.7.1 32b.

import numpy as np
from numpy import genfromtxt
import cv2 as cv

data = genfromtxt("DATA/bank_note_data.txt", delimiter=",")

# By  convention Feature = X, Labels = y

y = labeles= data[:,4]

X= features = data[:,0:4]

from sklearn.model_selection import train_test_split

# The train_test_split function will automatically split the X and y into 70% training, 30% test data
# It will also shuffle the data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print (X_train.min(),X_train.max()) # => min = -13.2869, max = 17.1116

# In this is case the range of the data set is from -13 to 17, which is not bad
# But there are times when it could go from a small -ve to large +ve val eg: -0.5 to 10^6
# in such cases you need to use a min max scalar.

from sklearn.preprocessing import MinMaxScaler
scalar_obj = MinMaxScaler()
scalar_obj.fit(X_train)

MinMaxScaler(copy=True, feature_range= (0,1))

scalar_X_train = scalar_obj.transform(X_train)
scalar_X_test = scalar_obj.transform(X_test)

# Now the train/test data set ranges from 0-1
# print (scalar_X_train.min(),scalar_X_train.max()) # => min = 0, max = 1

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4, input_dim=4, activation='relu'))
    
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

model.fit(scalar_X_train, y_train, epochs=50, verbose=2)

predictions = model.predict_classes(scalar_X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

model.save("bank_note.h5")

from keras.models import load_model

my_model = load_model("bank_note.h5")

predictions_new =my_model.predict_classes(X_test)

print(confusion_matrix(y_test, predictions_new))
print(classification_report(y_test, predictions_new))
