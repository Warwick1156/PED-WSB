import pandas as pd
import numpy as np

from multiprocessing import cpu_count, Pool


def parallelize_dataframe(df: pd.DataFrame, func, n_cores=cpu_count()):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    return df

