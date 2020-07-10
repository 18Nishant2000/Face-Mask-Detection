from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import os
import cv2
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


categories = [
    'with mask',
    'without mask'
]

l = LabelEncoder()
numerical = l.fit_transform(categories)

print(categories)
print(numerical)

X_train = []
Y_train = []
X_test = []
Y_test = []
size = (100, 100)

for i in range(len(categories)):
    images = os.listdir(f'dataset/{categories[i]}')
    train_size = int(len(images)*.8)
    for j in range(train_size):
        mat = cv2.imread(f'dataset/{categories[i]}/{images[j]}', cv2.IMREAD_GRAYSCALE)
        mat = cv2.resize(mat, size)
        X_train.append(mat)
        Y_train.append(numerical[i])
    for k in range(len(images[train_size:])):
        mat = cv2.imread(f'dataset/{categories[i]}/{images[j]}', cv2.IMREAD_GRAYSCALE)
        mat = cv2.resize(mat, size)
        X_test.append(mat)
        Y_test.append(numerical[i])

X_train_len = len(X_train)
X_test_len = len(X_test)
X_train = np.array(X_train)
X_test = np.array(X_test)

# X_train = X_train/255
# X_test = X_test/255

X_train = X_train.reshape(X_train_len, 100, 100, 1)
X_test = X_test.reshape(X_test_len, 100, 100, 1)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(100, 100, 1)),
    MaxPool2D(2, 2, padding='valid'),
    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPool2D(2, 2, padding='valid'),
    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPool2D(2, 2, padding='valid'),
    Conv2D(32, 3, activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=1)
model.summary()

model.save('saved_model2')

pred = model.predict(X_test)
print(pred)
