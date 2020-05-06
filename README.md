# Digit Recognizer
Convolutional Neural Network created using Keras with tensorflow backend. This neural network is used to predict which digit is on a photo.

## Training data

Google's SVHN was used to train this model.

## Input

Input to the model is a 32x32 image conataining one digit.

<p align="center">
  <img src="./Plots/Original_Photo.png">
</p>

## Model
Model uses Conv2D layers, along with Dropuot, MaxPooling2D, Flatten and dense layers to categorize photos into categories each representing one digit.

<p align="center">
  <img src="./Plots/Network_diagram.png">
</p>

## Output
Models output is a digit from 0 to 9 which represents the digit recognized on the photo

<p align="center">
  <img src="./Plots/Predictions.png">
</p>

## Model evaluation

| Accuracy  | 94% |
|-----------|-----|
| Recall    | 93% |
| Precision | 93% |
| F1 Score  | 94% |

<p align="center">
  <img src="./Plots/Confusion_Matrix.png">
</p>

<p align="center">
  <img src="./Plots/Receiver_Operating_Characteristic.png">
</p>
