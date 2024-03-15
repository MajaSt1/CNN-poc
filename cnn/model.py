import tensorflow as tf
from sklearn.model_selection import train_test_split

# Wczytanie danych CIFAR-10
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Podział danych na zbiory treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

X_train_cnn = tf.keras.utils.normalize(X_train, axis=1)
X_test_cnn = tf.keras.utils.normalize(X_test, axis=1)
X_val_cnn = tf.keras.utils.normalize(X_val, axis=1)

import matplotlib.pyplot as plt


def draw_curves(history, key1='accuracy', ylim1=(0.7, 1.00)):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')

    plt.show()


# Definicja modelu CNN
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=5,
                          verbose=1)

# Trening modelu
history = model.fit(X_train_cnn,
                    y_train,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test_cnn, y_test),
                    callbacks=[EarlyStop],
                    )

draw_curves(history, key1='accuracy')

score = model.evaluate(X_val_cnn, y_val, verbose=0)
print("CNN Error: %.2f%%" % (100 - score[1] * 100))
