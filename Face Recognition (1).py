#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


model=Sequential()


# In[3]:


model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))


# In[4]:


model.add(MaxPooling2D(pool_size = (2, 2)))


# In[5]:


model.add(Flatten())


# In[29]:


model.add(Dense(init="uniform",activation="relu",output_dim=120))


# In[30]:


model.add(Dense(init="uniform",activation="relu",output_dim=100))


# In[31]:


model.add(Dense(init="uniform",activation="relu",output_dim=30))


# In[32]:


model.add(Dense(init="uniform",activation="sigmoid",output_dim=3))


# In[33]:


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


# In[34]:


from keras.preprocessing.image import ImageDataGenerator


# In[35]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[36]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[37]:


x_train = train_datagen.flow_from_directory(r'C:\Users\Saif\face recognition\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                     class_mode = 'categorical')


# In[38]:


x_test = test_datagen.flow_from_directory(r'C:\Users\Saif\face recognition\testing_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[39]:


print(x_train.class_indices)


# In[40]:


x_train[0]


# In[42]:


model.fit_generator(x_train,
                         steps_per_epoch = 51,
                         epochs = 20,
                         validation_data = x_test,
                         validation_steps = 9)


# In[43]:


model.save("face recognition.h5")


# In[44]:


from keras.models import load_model


# In[45]:


import cv2


# In[46]:


import numpy as np


# In[47]:


model2=load_model("face recognition.h5")


# In[48]:


model2.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


# In[49]:


from skimage.transform import resize


# In[50]:


def detect(frame):
    try:
        img = resize(frame,(64,64))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        prediction = model.predict(img)
        print(prediction)
        prediction = model.predict_classes(img)
        print(prediction)
    except AttributeError:
        print("shape not found")


# In[51]:


frame=cv2.imread(r"C:\Users\Saif\face recognition\testing_set\hrithik roshan\2Q__ (60).jpg")


# In[52]:


data=detect(frame)


# In[ ]:




