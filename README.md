# Plant Disease Detection using InceptionV3

This repository contains a project for detecting plant diseases using a convolutional neural network (CNN) based on the InceptionV3 architecture. The goal is to classify images of plant leaves into different categories of diseases.

# Table of Contents
1.Installation
2.Dataset
3.Model Architecture
4.Training
5.Evaluation
6.Usage
7.Results

# Ensure you have the following libraries installed:

TensorFlow
NumPy
Matplotlib

# Dataset
The dataset used for this project should be organized in the following structure:

Plant Disease Data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── valid/
    ├── class1/
    ├── class2/
    └── ...
train/ directory contains the training images, categorized by disease type.
valid/ directory contains the validation images, also categorized by disease type.

# Model Architecture
The model is based on the InceptionV3 architecture, pre-trained on ImageNet. We add a few layers on top to adapt it to our classification task.

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

IMAGE_SIZE = [224, 224]

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in inception.layers:
    layer.trainable = False

x = Flatten()(inception.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=inception.input, outputs=prediction)

# Training
To train the model, run the following script:

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Plant Disease Data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('Plant Disease Data/valid', target_size=(224, 224), batch_size=32, class_mode='categorical')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

r = model.fit(training_set, validation_data=test_set, epochs=30, steps_per_epoch=len(training_set), validation_steps=len(test_set))
# Evaluation
To evaluate the model, use the following code:


import matplotlib.pyplot as plt

# Plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
![image](https://github.com/Kavin302004/Leaf_disease_Detection/assets/140266232/a9fd35bb-e2f4-4e34-bb18-d6093d89dcd8)

# Plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
![image](https://github.com/Kavin302004/Leaf_disease_Detection/assets/140266232/a1a0bee2-4839-4d96-8fb1-98c447e50d56)

# Usage
To use the trained model for prediction, run the following code:

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model_inception.h5')

img = image.load_img('path_to_image.jpg', target_size=(224, 224))
x = image.img_to_array(img) / 255
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
pred_class = np.argmax(pred, axis=1)
print(pred_class)

# Results
The results of the training and validation can be visualized through the loss and accuracy plots. The model's predictions can be compared against the actual classes to evaluate its performance.
