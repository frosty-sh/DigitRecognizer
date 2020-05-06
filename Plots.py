import numpy as np
import h5py
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve,auc

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
        
# Load data
h5f = h5py.File("./h5data/SVHN_grey.h5", 'r')
X_val = h5f['x_val'][:]
y_val_ohc = h5f['y_val'][:]
y_val = np.argmax(y_val_ohc,axis=1)
y_train = h5f['y_train'][:]

# Load model
with open('./vars/model', 'rb') as file:
    model = pickle.load(file)
    
# Plot training data distribution
plt.hist(np.argmax(y_train,axis=1))
plt.xlabel('class')
plt.ylabel('samples')
plt.title('training set class distribution')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.show()

# Plot training data distribution
plt.hist(y_val)
plt.xlabel('class')
plt.ylabel('samples')
plt.title('evaluation set class distribution')
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.show()
    
# Load training history
with open('./vars/training_data', 'rb') as file:
    training = pickle.load(file)

# Plot loss
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='bottom right')
plt.show()

# Plot accuracy
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='bottom right')
plt.show()

# Evaluate model
evaluation = model.evaluate(X_val, y_val_ohc, batch_size=512)
# print(evaluation)

# Predict
predictions_ohc = model.predict(X_val)
predictions = np.argmax(predictions_ohc,axis=1)
# Plot images
plot_images(X_val,predictions,2,5)
# Create matrix
con_mat = confusion_matrix(y_val,predictions)
# Plot matrix
plot_confusion_matrix(conf_mat=con_mat, 
                      figsize=(7, 7),
                      show_absolute=False,
                      show_normed=True,
                      class_names=[0,1,2,3,4,5,6,7,8,9])
plt.show()
print(classification_report(y_val,predictions))

# Reciever operating characteristic curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,10):
    fpr[i], tpr[i], _ = roc_curve(y_val_ohc[:, i], predictions_ohc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0,10)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(0,10):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= 10
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["macro"]))
plt.title('receiver operating characteristic curve')
plt.xlabel('average true positive rate')
plt.ylabel('average alse positive rate')
plt.legend(loc="lower right")
plt.show()
print(roc_auc["macro"])

#First and second convolutional layer
test_img=X_val[5,:,:,0]
plt.imshow(test_img)
plt.show()
kernel1_1 = model.layers[0].get_weights()[0][:,:,0,0]
plt.imshow(kernel1_1)
plt.title('first kernel')
plt.show()
result = np.zeros(test_img.shape)
for ii in range(test_img.shape[0] - 3):
    for jj in range(test_img.shape[1] - 3):
        result[ii, jj] = (test_img[ii:ii+3, jj:jj+3] * kernel1_1).sum()
plt.imshow(result)
plt.show()
kernel2_2 = model.layers[1].get_weights()[0][:,:,0,1]
plt.imshow(kernel2_2)
plt.title('second kernel')
plt.show()
result2 = np.zeros(result.shape-3)
for ii in range(result.shape[0] - 3):
    for jj in range(result.shape[1] - 3):
        result2[ii, jj] = (result[ii:ii+3, jj:jj+3] * kernel2_2).sum()
plt.imshow(result2)
plt.show()