import os
import os.path
import numpy as np
import glob
from pathlib import Path
from helper import normalize_data, read_data


if __name__ == "__main__":

	dataset = 'ShapeNet' 
	category = 'airplane'
	mode = 'test'
	root = '/path/to/original/dataset/'
	dump_root = '/path/to/Data/' + dataset + '/'

	if dataset not in ['ModelNet10', 'ShapeNet', 'dfaust', 'faces', 'sunrgbd']:
		raise Exception('dataset error.') 

	if dataset == 'ShapeNet':
		data_dir = root + mode + '_data/' + category + '/'
		list_el = glob.glob(os.path.join(data_dir, '*.pts'))
	elif dataset == 'ModelNet10':
		data_dir = root + category + '/' + mode + '/' 
		list_el = glob.glob(os.path.join(data_dir, '*.off'))
	
	dump_dir = dump_root + mode + '_data_npy/' + category + '/' 

	if not os.path.exists(dump_dir):
	    os.makedirs(dump_dir)

	for i, name in enumerate(list_el):
 
	    pc = read_data(name, dataset)
	    pc = normalize_data(pc)

	    file_name = Path(name).stem 
	    np.save(os.path.join(dump_dir, file_name + '.npy'), pc)	

	print('done!')

