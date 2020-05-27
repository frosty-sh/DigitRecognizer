# Content:
# Transforming input image
# Finding contours
# Filtering contours
# Classifying contours
# Prepraing images for prediction model
# Prediction

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle

path = "./Prediction images/Decimal numbers/im1.jpg"
img_color = Image.open(path)
img_color = np.array(img_color)

# region Transforming input image

# Convertign image to grayscale
img_gray = cv2.imread(path, 0)

# Creating two possible values for each pixel, 0 or 255, and assigning it based on treshold=100
th, img_filtered = cv2.threshold(
    img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# endregion

# region Finding contours

im2, contours, hierarchy = cv2.findContours(
    img_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# endregion

# region Filtering contours

areas = []
cnt = []

# Appending contours whose parent is equal to -1 (doesn't exist) or 0 (frame contour)
for i, c in enumerate(contours):
    if(hierarchy[0][i][3] < 1):
        cnt.append(c)
        areas.append(cv2.contourArea(c))

# Popping the largest contour which is the image frame
ctn_frame_index = areas.index(max(areas))
cnt.pop(ctn_frame_index)
areas.pop(ctn_frame_index)

# endregion

# region Classifying contours

cnt_numbers = []
cnt_rest = []

# Numbers detection
for a in areas:
    if a > (max(areas) - max(areas)/1.5):
        cnt_numbers.append(cnt[areas.index(a)])
    else:
        cnt_rest.append(cnt[areas.index(a)])

cnt_numbers_x = []
cnt_rest_x = []

for i in range(len(cnt_numbers)):
    cnt_numbers_x.append(cnt_numbers[i][0][0][0])

for i in range(len(cnt_rest)):
    cnt_rest_x.append(cnt_rest[i][0][0][0])


cnt_dot = []
cnt_dot_x = []

# Dot detection
for i in range(len(cnt_rest)):
    if(min(cnt_numbers_x) < cnt_rest_x[i] < max(cnt_numbers_x)):
        cnt_dot = cnt_rest[i]
        cnt_dot_x = cnt_rest_x[i]
        break


cnt_numbers_left = []
cnt_numbers_right = []

# Determining number position relative to dot (decimal separator)
for i, c in enumerate(cnt_numbers):
    if cnt_numbers_x[i] > cnt_dot_x:
        cnt_numbers_right.append(c)
    else:
        cnt_numbers_left.append(c)


# Creating boundary boxes around dot
x, y, w, h = cv2.boundingRect(cnt_dot)
cv2.rectangle(img_color, (x, y), (x+w, y+h), (100, 100, 50), 1)

plt.imshow(img_color)
plt.show()

# endregion

# region Preparing images for prediction model


def get32x32images(contour):
    images_return = []

    for i, c in enumerate(contour):
        x, y, width_1, height_1 = cv2.boundingRect(c)
        img = img_filtered[y:y+height_1, x:x+width_1]

        plt.imshow(img)
        plt.show()

        # Resize image - height to 32py, width is resized proportionally
        height_2 = 32
        width_2 = height_2 * width_1 / height_1
        img = cv2.resize(img, (int(width_2), height_2))

        # Expand image width with white pixels
        rest_space = 32 - int(width_2)

        if rest_space > 0:
            # Equal space distribution on each side
            left_space = int(rest_space / 2)
            right_space = rest_space - left_space

            np_left_space = np.ones((height_2, left_space))
            np_right_space = np.ones((height_2, right_space))

            np_left_space[np_left_space < 255] = 255
            np_right_space[np_right_space < 255] = 255

            # Appending both sides to image
            img = np.append(img, np_right_space, 1)
            img = np.append(np_left_space, img, 1)

            plt.imshow(img)
            plt.show()

        # Creating extra dimension for image channel
        img = np.expand_dims(img, axis=-1)
        images_return.append(img)

    return np.asarray(images_return)


imgs_num_left = get32x32images(cnt_numbers_left)
imgs_num_right = get32x32images(cnt_numbers_right)

# endregion

# region Prediction

with open('./vars/model', 'rb') as file:
    model = pickle.load(file)

left_preds = model.predict(imgs_num_left)
right_preds = model.predict(imgs_num_right)

print("Predicted number: ", end='')

for p in np.flipud(left_preds):
    print(str(np.argmax(p)), end='')

print(".", end='')

for p in np.flipud(right_preds):
    print(str(np.argmax(p)), end='')

# endregion
