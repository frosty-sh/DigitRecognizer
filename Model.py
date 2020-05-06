from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import numpy as np
import h5py

# Fetch data
h5f = h5py.File("./h5data/SVHN_grey.h5", 'r')
X_train = h5f['x_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['x_test'][:]
y_test = h5f['y_test'][:]

# Create sequential model
model = Sequential([
    Conv2D(32, kernel_size=9, activation='relu', input_shape=(32, 32, 1)),
    Conv2D(32, kernel_size=9, activation='relu'),
    MaxPooling2D(2, strides=2),
    Dropout(0.3),
    Conv2D(64, kernel_size=9, activation='relu'),
    Conv2D(64, kernel_size=9), activation='relu'),
    MaxPooling2D(2, strides = 2),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
training = model.fit(X_train, y_train, epochs=10,
                     batch_size=512, validation_data=(X_test, y_test))

# Export model
import pickle
with open('./vars/training_data', 'wb') as file:
    pickle.dump(training, file)
with open('./vars/model', 'wb') as file:
    pickle.dump(model, file)
model.save('model.h5')