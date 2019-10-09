import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, InputLayer
from keras import backend as K
from os import path
from distutils.version import LooseVersion as LV
from keras import __version__

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers import SGD
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

output = 'outputs/mlp_multiclass/'
# Data preprocessing
if not path.exists(output + 'X.npy'):
    vector_path = 'outputs/subject_classification_dataset_0.txt'
    v = np.loadtxt(vector_path, delimiter=',', dtype='str', encoding='utf8')
    np.random.shuffle(v)
    # v = v[:100,:]
    X = v[:, :-1].astype('float')
    Y = v[:, -1]
    Y = np.array(pandas.get_dummies(Y))
    # Y = Y.values.argmax(1)
    np.save(output + 'X', X)
    np.save(output + 'Y', Y)
else:
    X = np.load(output + 'X.npy')
    Y = np.load(output + 'Y.npy')

# Y = np.array(Y)
# print(Y)
# print(Y.shape)
Y = Y[:, 1:]
# X = X[:, :300]
X = preprocessing.scale(X)
# print(X)
# print(Y)
# print(X.shape)
# print(Y.shape)
# X = X[93145:93245, :]
# Y = Y[93145:93245]
# Y = (Y > 10).astype(int)
record_count = X.shape[0]
class_count = Y.shape[1]
input_size = X.shape[1]
train_size = int(0.2 * record_count)
X_train = X[:train_size, :]
X_train = X_train.astype('float32')
X_test = X[train_size:, :]
Y_train = Y[:train_size, :]
Y_test = Y[train_size:, :]

print()
print('data loaded: train:', len(X_train), 'test:', len(X_test))

# Model architecture
model = Sequential()
model.add(InputLayer(input_shape=(input_size,)))
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(Dense(units=40))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
model.add(Dense(units=class_count, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())

# Learning
epochs = 20  # one epoch with simple model takes about 4 seconds

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=4,
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
