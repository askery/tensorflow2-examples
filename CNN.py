""" Simple CNN implementation with TensorFlow 2.0 for MNIST dataset.
Author: Askery Canabarro 
Tested with TF 2.3.0
"""

# STEP 0: Import TensorFlow and dependencies/libraries.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
print ("TF version: ", tf.__version__)

# STEP 1: Load the MNIST dataset.
(X_train,y_train),(X_test,y_test) =tf.keras.datasets.mnist.load_data()

# STEP 1a: Inspect type and shapes.
print ("Data type     ", type(X_train)) # numpy array. Nice!
print ("Xtrain shape: ", X_train.shape) # 60000 examples, 28 x 28 = 784 shape
print ("Xtest shape:  ", X_test.shape)  # 10000 examples, 28 x 28 = 784 shape
print ("ytrain shape: ", y_train.shape)
print ("ytest  shape: ", y_test.shape)



# STEP 1b: EDA
# print(X_train[0]), print(y_train[0])  # raw data

plt.imshow(X_train[0], cmap='gray')     # see an example of the data
plt.axis('off')
plt.show()

# STEP 1c: Preprocessing. 

# Rescale the images pixels from range [0,255] to range [0.0,1.0].
X_train = X_train/255
X_test  = X_test/255

# TensorFlow likes cube of data!!!
X_train = X_train.reshape((-1,28,28,1)) # 60000 instances with 28x28x1 shape
X_test  = X_test.reshape((-1,28,28,1))  # 10000 instances with 28x28x1 shape




#pd.DataFrame(y_train).hist(bins = 50)

#pd.DataFrame(y_test).hist(bins = 50)

#pd.DataFrame(y_test)[0].unique()  #outra forma de analisar

# STEP 2: build the model (functional API)
K = len (set(y_train) )             # number of classes
i = Input(shape = X_train[0].shape) # 28x28x1: all inputs have the same shape, get the shape of the first one.
x = Conv2D(  32, [3,3], activation='relu')(i)
x = Conv2D(  64, [3,3], activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(K,activation='softmax')(x)

model = Model(i,x)

print(model.summary())

# optimization and loss parameters
opt  = "adam"
cost = "sparse_categorical_crossentropy"
model.compile(optimizer=opt,
                   loss=cost,
                metrics=['accuracy'])

# STEP 3: Train the model.

r = model.fit(X_train,y_train, 
          validation_data=(X_test,y_test), 
          epochs=10)

# STEP 3a: See loss and accuracy in function of epochs
#
plt.plot(r.history['loss'], label ='train loss')
plt.plot(r.history['val_loss'], label ='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
#
plt.plot(r.history['accuracy'], label ='train acc')
plt.plot(r.history['val_accuracy'], label ='val acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.show()

# STEP 4: Evaluate your model.
print("Test acc:", model.evaluate(X_test,y_test)[0])


inv  = model.predict(X_test)
inv  = np.array ( list (map(np.argmax, inv)) ) #não precisa entender agora: transformando em inteiro classificador


# STEP 4a: Deeper evaluate your model.
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
clas   = list(map(str,range(0,10)))
preds  = model.predict(X_test)
P      = np.array ( list (map(np.argmax, preds)) )
cm     = confusion_matrix(P,y_test)
df_cm  = pd.DataFrame(cm, index = clas, columns = clas, dtype='int32')

import seaborn as sn
plt.figure(figsize=(9,7))
sn.set(font_scale=1.4) 
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}) 
plt.show()

# STEP 4b: even deeper evaluation. Check misclassification example. Is it acceptable?
ind_err = np.array ( np.where(P != y_test) ).flatten() # getting indices where prediction differs from actual class
ind     = np.random.choice(ind_err)
plt.imshow(X_test[ind].reshape((28,28)), cmap = "gray")
plt.title ("Pred: " + str(clas[P[ind]]) + ", True: " + str(clas[y_test[ind]]))
plt.grid(False)
plt.show()