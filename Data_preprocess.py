# Content
# https://github.com/aditya9211/SVHN-CNN/blob/master/data_preprocess.ipynb
# Create a Startfied 13% of data in Validation Set
# Convertih Label 10's to 0's
# Greyscale conversion of image for easy computation
# Normalization of data
# One hot label encoding

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

plt.rcParams['figure.figsize'] = (16.0, 4.0)


# region loading and transposing .mat files
def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']


X_train, y_train = load_data('./SVHN/train_32x32.mat')
X_test, y_test = load_data('./SVHN/test_32x32.mat')


# Transposing the the train and test data by converting it from
# (width, height, channels, size) -> (size, width, height, channels)

X_train, y_train = X_train.transpose((3, 0, 1, 2)), y_train[:, 0]
X_test, y_test = X_test.transpose((3, 0, 1, 2)), y_test[:, 0]

print("--Transposed--")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# endregion


def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i, :, :, 0], cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[i])


def plot_image(img, label):
    fig, ax = plt.subplots()
    ax.imshow(img,  cmap=plt.get_cmap('gray'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label)


plot_images(X_train, y_train, 5, 5)
plt.show()
# region converting 10's to 0 's

print(np.unique(y_train))
print(np.unique(y_test))

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0


print(np.unique(y_train))
print(np.unique(y_test))
plot_images(X_test, y_test, 2, 8)
plt.show()
# endregion

# region splitting the training to train+validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.13, random_state=7)
# endregion

# region ploting distribution of data - train - test - val

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex="all")
fig.suptitle("class distribution", fontsize=14, fontweight="bold", y=1.05)


ax1.hist(y_train, bins=10)
ax1.set_title("train set " + str(y_train.shape))
ax1.set_xlim(1, 10)

ax2.hist(y_test, color="g", bins=10)
ax2.set_title("test set " + str(y_test.shape))
ax2.set_xlim(1, 10)

ax3.hist(y_val, color="r", bins=10)
ax3.set_title("validation set set " + str(y_val.shape))
ax3.set_xlim(1, 10)

plt.show()

# endregion

# region grayscale conversion


def rgb2gray(images):
    # y = 0.2990r + 0.5870g + 0.1140b
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)
    # return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)
val_greyscale = rgb2gray(X_val).astype(np.float32)


# region testing single image

# image, label = x_train[8], y_train[8]
# plot_image(image, label)

# imagegray = np.dot(image, [0.2989, 0.5870, 0.1140])
# plot_image(imagegray, label)

# endregion

print("training set", train_greyscale.shape)
print("validation set", val_greyscale.shape)
print("test set", test_greyscale.shape)


# plot_images(train_greyscale, y_train, 1, 3)
# plot_images(test_greyscale, y_test, 1, 3)
# plot_images(val_greyscale, y_val, 1, 3)

# endregion

# region normalization

# process of the normalizing data dimensions so that they are of approximetely the same scale. divide each dimension by its standard deviation once it has been zero-centered

train_mean = np.mean(train_greyscale, axis=0)
train_std = np.std(train_greyscale, axis=0)

# normalizing data
train_greyscale_norm = (train_greyscale - train_mean) / train_std
test_greyscale_norm = (test_greyscale - train_mean) / train_std
val_greyscale_norm = (val_greyscale - train_mean) / train_std


# plot_images(train_greyscale_norm, y_train, 1, 3)
# plot_images(test_greyscale_norm, y_test, 1, 3)
# plot_images(val_greyscale_norm, y_val, 1, 3)

# endregion

# region one hot label encoding

enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

# endregion

plot_images(X_test, y_test, 3, 3)

# region storing data to disk
h5f = h5py.File('svhn_grey.h5')
h5f.create_dataset("x_train", data=train_greyscale_norm)
h5f.create_dataset("y_train", data=y_train)
h5f.create_dataset("x_test", data=test_greyscale_norm)
h5f.create_dataset("y_test", data=y_test)
h5f.create_dataset("x_val", data=val_greyscale_norm)
h5f.create_dataset("y_val", data=y_val)
h5f.close()

# endregion
plt.show()
