import os
import gc
import pickle
import keras_ocr
import cv2 as cv

from data import path
from tqdm import tqdm 
from glob import glob
from math import ceil

def _resize_to_max_size(img, size):
    height, width, z = img.shape
    if height >= width: 
        if height > size:
            dim = size, ceil(width * (1.0 / (height / size)))
            return cv.resize(img, dim)
        else:
            return img
    else:
        if width > size:
            dim = ceil(height * (1.0 / (width / size))), size
            return cv.resize(img, dim)
        else:
            return img
        

def _prepare_img(image_path):
    res = cv.imread(image_path)

    # res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    return _resize_to_max_size(res, 384)

def process_images(paths: list, batch_size=5):  
    pipeline = keras_ocr.pipeline.Pipeline()
    
    results = []
    for i in tqdm(range(ceil(len(paths) / batch_size))):
        gc.collect()
        x, x_2 = i*(batch_size), (i+1) * batch_size
        print("++++++++++++++++++++++++++++++++++++++++++++++++++", x, x_2)
        paths_batch = paths[x:x_2]
        
        filtrated_paths = [path_ for path_ in paths_batch if path_.split('.')[-1] in ['jpg', 'png']]
        posts_id = [os.path.basename(path_.split('.')[-2]) for path_ in filtrated_paths]

        images = [_prepare_img(image) for image in filtrated_paths]

        for img in images:
            print("IMAGE SIZE", img.shape)
        
        ocr_results = pipeline.recognize(images)
        texts = [[tuple_[0] for tuple_ in result if len(tuple_[0]) > 1] for result in ocr_results]
        
        result = list(zip(posts_id, texts))
        for item in result:
            results.append(item)

        with open(os.path.join(path('temp'), str(i) + '_ocr.pkl'), 'wb') as f:
            pickle.dump(results, f)
            print('Autosaved')
            
    return results
	

if __name__ == '__main__':
	list_of_image_files = glob(os.path.join(path('image'), '*'))
	result = process_images(list_of_image_files)
	
	with open(os.path.join(path('data'), 'ocr.pkl'), 'wb') as f:
            pickle.dump(result, f)
            print('Done')
