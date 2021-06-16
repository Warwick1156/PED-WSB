import os
import sys
src_dir = os.path.join('..', 'src')
sys.path.append(os.path.abspath(src_dir))
 
import pandas as pd
import numpy as np
import seaborn as sns
 
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
 
from data import path, get_dataset
 
data = get_dataset('comments_v2.csv')
 
start = '2021-01-19 11:00:00'
timeformat = '%Y-%m-%d %H:%M:%S'
begining = datetime.strptime(start, timeformat)
 
def get_datetime(unix_time: list):
    return [datetime.utcfromtimestamp(ts).strftime(timeformat) for ts in unix_time]
 
data['datetime'] = get_datetime(data.created_utc.values)
 
values = 4320
l = [begining + timedelta(minutes = 10 * i) for i in range(values)]
timeseries = pd.DataFrame()
timeseries['datetime'] = [x.strftime(timeformat) for x in l]
 
timeseries['datetime'] = pd.to_datetime(timeseries['datetime'])  
data['datetime'] = pd.to_datetime(data['datetime'])  
 
def n_comments(timeseries, df):
    result = []
    for i in range(len(timeseries)):
        try:
            mask = (df['datetime'] >= timeseries.datetime[i]) & (df['datetime'] < timeseries.datetime[i + 1])
            result.append(df[mask].shape[0])
        except:
            result.append(0)
        
    return result
 
timeseries['comments'] = n_comments(timeseries, data)
 
plt.figure(figsize=(20, 10))
sns.lineplot(data=timeseries[:3500], x="datetime", y="comments")
 
def create_timeseries(comments, shape):
    post_ids = comments.post_id.unique()
    
    timeseries_df = pd.DataFrame({
        'post_id': [],
        'datetime': [],
        'new_comments': [],
        'texts': []
    })
    
    for id_ in tqdm(post_ids):
        post_comments = comments[comments.post_id == id_]
        post_created_time = post_comments.datetime.values[0]
        
        times = [post_created_time]
        result = []
        texts = []
        for i in range(shape):
            start_time = times[-1]
            end_time = start_time + np.timedelta64(10, 'm')
            times.append(end_time)
            
            mask = (post_comments['datetime'] >= start_time) & (post_comments['datetime'] < end_time)
            result.append(post_comments[mask].shape[0])
            
            derp = post_comments[mask]
            text = list(derp.body.values)
            for _ in range(10 - len(text)):
                text.append('')
                
            texts.append(text)
        
        result[0] = result[0] - 1
        
        temp_df = pd.DataFrame({
            'post_id': [id_] * shape,
            'datetime': times[-2],
            'new_comments': result,
            'texts': texts
        })
        
        timeseries_df = timeseries_df.append(temp_df, ignore_index=True)
        
    return timeseries_df
 
shape = 24 * 2 * 6
my_timeseries = create_timeseries(data, shape)
my_timeseries