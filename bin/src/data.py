import pandas as pd
import json
import os

from os import path
from glob import glob


def _load_json() -> dict:
	with open(os.path.join('..', 'src', 'settings.json')) as f:
		return json.load(f)


def path(target: str) -> str:
	settings = _load_json()
		
	paths_dict = settings['path'][0]	
	raw_path = paths_dict[target]
	
	path_ = ''
	for part in raw_path:
		path_ = os.path.join(path_, part)
		
	return path_
	
	
def get_dataset(target=None, separator='`', verbose=True) -> pd.DataFrame:
	datasets_dir = path('dataset')
	
	if target is None:	
		file_list = glob(os.path.join(datasets_dir, '*.csv'))
		path_ = max(file_list, key=os.path.getctime)
	else:
		path_ = os.path.join(datasets_dir, target)
		
	if verbose:
		print(path_)
		
	return pd.read_csv(path_, sep=separator)
