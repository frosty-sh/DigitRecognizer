import numpy as np
from PIL import Image
import pickle


def rgb2gray(images):
    # y = 0.2990r + 0.5870g + 0.1140b
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)
    # return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


eval = np.array(Image.open('./eval2.jpg'))
eval = np.expand_dims(eval, axis=-1)
eval = np.array([eval])

with open('./vars/model', 'rb') as file:
    model = pickle.load(file)

prediction = model.predict(eval)

print(np.argmax(prediction,axis=1))