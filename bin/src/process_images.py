#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np


# In[ ]:





# In[14]:


data_dir = os.path.join('..', '..', 'data')
img_dir  = os.path.join('..', '..', 'data', 'img')
temp_dir = os.path.join('..', '..', 'data', 'temp')


# In[ ]:





# In[3]:


predicate = lambda f: os.path.isfile(os.path.join(img_dir, f)) and f[-3:] in ['png', 'jpg']
img_files = [f for f in os.listdir(img_dir) if predicate(f)]


# In[11]:


from skimage.color import rgb2hsv, hsv2rgb
from skimage import data, io
from matplotlib import pyplot as plt

# x = np.array([[[255,0,0] for _ in range(3)] for _ in range(3)])

# # io.imshow(x)
# # plt.show()
# x = rgb2hsv(x)
# x[0,0] = [1, 1, 250]
# x = hsv2rgb(x)
# io.imshow(x)
# plt.show()
# np.around(x[:,:,0], 2)


# In[5]:


from keras.applications.vgg19  import VGG19, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array

model = VGG19()


# In[ ]:





# In[6]:


# from PIL import Image, ImageDraw, ImageFont #dynamic import

# def gif_to_png(image_path):
#     Image.open(image_path).save(image_path[:-3] + "png",'png', optimize=True, quality=70)


# In[7]:


# predicate = lambda f: os.path.isfile(os.path.join(img_dir, f)) and f[-3:] in ['gif']
# img_files = [f for f in os.listdir(img_dir) if predicate(f)]

# for i in tqdm(range(len(img_files[:3]))):

#     img_path = os.path.join(img_dir, img_files[i])
    
#     print(img_path)
#     gif_to_png(img_path)


# In[8]:


def get_img_label(image):
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat  = model.predict(image)
    label = decode_predictions(yhat, top=5)
    return [label[0][i][1] for i in range(5)]


# In[9]:


import math

def get_img_avg_colors(image):

    X = 0.0
    Y = 0.0

    count = 0
    sat = 0
    val = 0
    
    hsv = rgb2hsv(image)

    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            
            X += math.cos(hsv[i,j,0] / 180.0 * math.pi)
            Y += math.sin(hsv[i,j,0] / 180.0 * math.pi)
            
            sat += hsv[i,j,1]
            val += hsv[i,j,2]
            count += 1

    X /= count
    Y /= count

    avg_hue = math.atan2(Y, X) * 180.0 / math.pi;
    avg_sat = sat / count
    avg_val = val / count / 255.0

    return [avg_hue, avg_sat, avg_val]


# In[16]:


from keras.preprocessing.image import img_to_array, load_img

from tqdm import tqdm

import pandas as pd

processed_data = []

for i in tqdm(range(len(img_files))):
    
    img_path = os.path.join(img_dir, img_files[i])
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    res = img_files[i:i+1] + get_img_avg_colors(image) + get_img_label(image)
    processed_data.append(res)
    
    if i % 500 == 0:
        print('Autosaving...')
        result = pd.DataFrame(processed_data, columns=['image_name', 'avg_hue', 'avg_saturation', 'avg_value', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
        result.to_csv(os.path.join(data_dir, "autosave_{}_image_traits.csv".format(i)), sep='`', index=False)
    
result = pd.DataFrame(processed_data, columns=['image_name', 'avg_hue', 'avg_saturation', 'avg_value', 'label_0', 'label_1', 'label_2', 'label_3', 'label_4'])
result.to_csv(os.path.join(data_dir, "image_traits.csv"), sep='`', index=False)


# In[ ]:


224/16


# In[ ]:





# In[ ]:





# In[ ]:




