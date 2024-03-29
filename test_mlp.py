from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, InputLayer
from keras import backend as K
from os import path
from distutils.version import LooseVersion as LV
from keras import __version__

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, metrics

sns.set()


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


print('Using Keras version:', __version__, 'backend:', K.backend())
assert (LV(__version__) >= LV("2.0.0"))

# Data preprocessing
if not path.exists('outputs/mlp/X.npy'):
    vector_path = 'outputs/v.txt'
    v = np.loadtxt(vector_path, delimiter=',')
    np.random.shuffle(v)
    X = v[:, :-1]
    Y = v[:, -1].astype('int')
    np.save('outputs/mlp/X', X)
    np.save('outputs/mlp/Y', Y)
else:
    X = np.load('outputs/mlp/X.npy')
    Y = np.load('outputs/mlp/Y.npy')

X = X[93145:93245, :]
Y = Y[93145:93245]
X = preprocessing.scale(X)
Y = (Y > 10).astype(int)
record_count = X.shape[0]
input_size = X.shape[1]
train_size = int(0.5 * record_count)
X_train = X[:train_size, :]
# X_train = X_train.astype('float32')
X_test = X[train_size:, :]
Y_train = Y[:train_size]
Y_test = Y[train_size:]

print()
print('data loaded: train:', len(X_train), 'test:', len(X_test))
print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)
print('+_rate:', len(np.where(Y_train == 1)[0]) / Y_train.shape[0])

print('Y_test:', Y_test.shape)
print('+_rate:', len(np.where(Y_test == 1)[0]) / Y_test.shape[0])

# Model architecture
model = Sequential()
model.add(InputLayer(input_shape=(input_size,)))
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc', f1_m, precision_m, recall_m])

###############################################
# # Model initialization:
# model = Sequential()
# model.add(InputLayer(input_shape=(input_size,)))
# # model.add(Flatten())
#
# # A simple model:
# model.add(Dense(units=20))
# model.add(Activation('relu'))
#
# # # A bit more complex model:
# # model.add(Dense(units=50))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.2))
#
# # model.add(Dense(units=50))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.2))
#
# # The last layer needs to be like this:
# model.add(Dense(units=1, activation='relu'))
#
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer='rmsprop',
#               metrics=['accuracy']
#               )
#################################################
# model = Sequential()
# model.add(Dense(input_dim=X_train.shape[1], output_dim=256))
# model.add(Activation("tanh"))
# model.add(Dropout(0.50))
# model.add(Dense(output_dim=128))
# model.add(Activation("relu"))
# model.add(Dropout(0.50))
# model.add(Dense(output_dim=64))
# model.add(Activation("relu"))
# model.add(Dropout(0.50))
# model.add(Dense(output_dim=1))
# model.compile("nadam", "mae")
#################################################

print(model.summary())

# Learning
epochs = 100  # one epoch with simple model takes about 4 seconds

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=80,
                    verbose=2)

# Curves
plt.figure(figsize=(5, 3))
plt.plot(history.epoch, history.history['loss'])
plt.title('loss')

model.save('outputs/mlp_model')
# plt.show()


# plt.figure(figsize=(5, 3))
# plt.plot(history.epoch, history.history['acc'])
# plt.title('accuracy')

# Inference
scores = model.evaluate(X_test, Y_test, verbose=2)
for i in range(1, len(scores)):
    print("%s: %.2f%%" % (model.metrics_names[i], scores[i] * 100))

# predictions = model.predict(X_test)
#
# show_failures(predictions)
