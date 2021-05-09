import os
import gc
import pickle
import keras_ocr
import cv2 as cv

from data import path
from tqdm import tqdm 
from glob import glob
from math import ceil

from data import path

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(image, dim, interpolation = inter)

    return resized
        

def _prepare_img(image_path, max_):
    img = cv.imread(image_path)

    height, width, _ = img.shape
    if height > width:
        res = image_resize(img, height=max_)
    else: 
        res = image_resize(img, width=max_)
        
    return res

def process_images(paths: list, batch_size: int=5, max_resolution: int=1024):  
    pipeline = keras_ocr.pipeline.Pipeline()
    
    results = []
    for i in tqdm(range(ceil(len(paths) / batch_size))):
        gc.collect()
        x, x_2 = i*(batch_size), (i+1) * batch_size
        print("++++++++++++++++++++++++++++++++++++++++++++++++++", x, x_2)
        paths_batch = paths[x:x_2]
        
        filtrated_paths = [path_ for path_ in paths_batch if path_.split('.')[-1] in ['jpg', 'png']]
        posts_id = [os.path.basename(path_.split('.')[-2]) for path_ in filtrated_paths]

        images = [_prepare_img(image, max_resolution) for image in filtrated_paths]

        for img in images:
            print("IMAGE SIZE", img.shape)
        
        try:
            ocr_results = pipeline.recognize(images)
            texts = [[tuple_[0] for tuple_ in result if len(tuple_[0]) > 1] for result in ocr_results]
            
            result = list(zip(posts_id, texts))
            for item in result:
                results.append(item)

            with open(os.path.join(path('temp'), str(i) + '_ocr.pkl'), 'wb') as f:
                pickle.dump(results, f)
                print('Autosaved')
        except:
            print('Something went wrong. Skiping batch')
        
            
    return results
    

def clear_temp():
    temp_files = glob(os.path.join(path('temp'), '*_ocr.pkl'))
    for ocr_temp_file in temp_files:
        os.remove(ocr_temp_file)
	

if __name__ == '__main__':
	list_of_image_files = glob(os.path.join(path('image'), '*'))
	result = process_images(list_of_image_files, 2)
	
	with open(os.path.join(path('data'), 'ocr.pkl'), 'wb') as f:
            pickle.dump(result, f)
            print('Done')
