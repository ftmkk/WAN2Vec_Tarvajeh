from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, InputLayer
from keras.utils import np_utils
from keras import backend as K

from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))



# Data preprocessing

nb_classes = 10

#X_train = X_train.astype('float32')
record_count = np.shape(tweet_embeddings)[1]
train_size = 0.8*record_count
X_train = tweet_embeddings[:,:train_size]
X_test =  tweet_embeddings[:,train_size:]

print()
print('MNIST data loaded: train:',len(X_train),'test:',len(X_test))
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Y_train:', Y_train.shape)

# Model architecture
# Model initialization:
model = Sequential()
model.add(InputLayer(input_shape=(768,)))
model.add(Flatten())

# A simple model:
model.add(Dense(units=20))
model.add(Activation('relu'))

# A bit more complex model:
#model.add(Dense(units=50))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

#model.add(Dense(units=50))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))

# The last layer needs to be like this:
model.add(Dense(units=300, activation='relu'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy','recall','precision'])
print(model.summary())


# Learning
epochs = 10 # one epoch with simple model takes about 4 seconds

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=32,
                    verbose=2)


# Curves
plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy')

# Inference
scores = model.evaluate(X_test, Y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# predictions = model.predict(X_test)
#
# show_failures(predictions)