#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import cv2
import os
from xml.etree import ElementTree
from matplotlib import pyplot as plt


# In[5]:


import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import datasets, layers, models
keras = tf.keras


# In[3]:


class_names = ['person','person-like']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

n_classes = 2
size = (200,200)


# In[4]:


def load_data():
    datasets = ['Train/Train', 'Test/Test', 'Val/Val']
    output = []
    for dataset in datasets:
        imags = []
        labels = []
        directoryA = "C:\\Users\\talaa\\OneDrive\\Desktop\\NN_Pro\\DataSet\\Train\\Train\\Annotations"
        directoryIMG = "C:\\Users\\talaa\\OneDrive\\Desktop\\NN_Pro\\DataSet\\Train\\Train\\JPEGImages"
        file = os.listdir(directoryA)
        img = os.listdir(directoryIMG)
        file.sort()
        img.sort()

        i = 0
        for xml in file:
            xmlf = os.path.join(directoryA, xml)
            dom = ElementTree.parse(xmlf)
            vb = dom.findall('object')
            label = vb[0].find('name').text
            labels.append(class_names_label[label])

            img_path = os.path.join(directoryIMG, img[i])
            if os.path.isfile(img_path):
                curr_img = cv2.imread(img_path)
                if curr_img is not None:
                    size = (200, 200)
                    curr_img = cv2.resize(curr_img, size)
                    imags.append(curr_img)
                else:
                    print(f"Error loading image {img_path}")
            else:
                print(f"Image file {img_path} not found")
            i += 1
        
        imags = np.array(imags, dtype='float32')
        imags = imags / 255
        
        labels = np.array(labels, dtype='int32')

        output.append((imags, labels))
    return output


# In[6]:


(train_images, train_labels),(test_images, test_labels),(val_images, val_labels) = load_data()


# In[7]:


train_images.shape


# In[8]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(train_images),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(train_images[i])  
    plt.title(class_names[train_labels[i]])
    plt.axis('off')


# In[13]:


model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[14]:


model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(2))


# In[15]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(train_images, train_labels, epochs=6, 
                    validation_data=(test_images, test_labels))


# In[34]:


def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


# In[35]:


plot_accuracy_loss(history)


# In[36]:


preds = model.predict(val_images) 


# In[20]:


plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(val_images),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(val_images[i])    
    plt.axis('off')
    x =np.argmax(preds[i]) # takes the maximum of of the 6 probabilites. 
    plt.title((class_names[x]))


# In[21]:


result = []
for i in range(len(preds)):
    result.append(np.argmax(preds[i]))


# In[22]:


tn, fp, fn, tp = confusion_matrix(val_labels,result).ravel()


# In[23]:


(tn, fp, fn, tp)


# In[24]:


model.summary()


# In[25]:


model.save_weights('C:\\Users\\talaa\\OneDrive\\Desktop\\NN_Pro\\checkpoints\\my_checkpoint')


# In[27]:


# !mkdir -p saved_model
model.save("C:\\Users\\talaa\\OneDrive\\Desktop\\NN_Pro\\my_model")


# In[ ]:




