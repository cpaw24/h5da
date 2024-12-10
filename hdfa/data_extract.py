
import h5py as h5
from zipfile import ZipFile
from typing import AnyStr, Dict, List
import numpy as np
from PIL import Image
import json
import csv



class DataExtractor:
	def __init__(self, output_file: AnyStr, input_file: AnyStr, input_dict: Dict | np.ndarray) -> None:
		self.input_file = input_file
		self.input_dict = input_dict
		self.output_file = output_file
		self.h5_file = h5.File(self.output_file, 'w', libver='latest',
		                       meta_block_size=8388608, locking=True, driver='core')

	def __parse_data(self, input_dict: Dict | np.ndarray) -> List:
		value_list: List = []
		if isinstance(input_dict, Dict):
			for k, v in input_dict.items():
				if isinstance(v, dict):
					self.__parse_data(v)
				elif isinstance(v, list):
					v = np.ndarray(v)
					value_list.append([k, v])
					continue
				elif isinstance(v, np.ndarray):
					value_list.append([k, v])
					continue
				elif isinstance(v, str):
					value_list.append([k, v])
					continue
		elif isinstance(input_dict, np.ndarray):
			value_list.append(input_dict)

		return value_list

	def __dataset_content(self, input_file: AnyStr) -> tuple[List, List]:
		processed_file_list: List = []
		with ZipFile(input_file) as datazip:
			file_list = datazip.namelist()
			for file in file_list:
				if file.endswith('.json'):
					processed_file_list.append(file)
					with datazip.open(file) as datafile:
						contents = datafile.read().decode('utf-8').rstrip().splitlines()
						return contents, processed_file_list
				elif file.endswith('.jpg' or '.jpeg' or '.png' or '.bmp' or '.tiff' or '.svg'):
					processed_file_list.append(file)
					with datazip.open(file) as datafile:
						img = Image.open(datafile)
						data = np.array(img)
						contents = self.__parse_data(data)
						return contents, processed_file_list
				elif file.endswith('.csv'):
					processed_file_list.append(file)
					with csv.reader(file, 'r', delimiter=",", quotechar="") as datafile:
						contents = datafile.read().decode('utf-8')
						return contents, processed_file_list

	def input_processor(self, input_file: AnyStr, h5_file: h5.File) -> h5.File.keys:
		content, file_list = self.__dataset_content(input_file)
		for line, file_nm in zip(content, file_list):
			file_group = file_nm.split('.')[0]
			h5_file.create_group(file_group)
			if isinstance(content, Dict):
				data = json.loads(line)
				kv_list = self.__parse_data(data)
				for kv in kv_list:
					k, v = kv
					if isinstance(v, np.ndarray):
						h5_file[f'{file_group}'].create_dataset(k, data=v, compression='gzip')
					elif isinstance(v, str):
						h5_file[f'{file_group}'].attrs[k] = v
			elif isinstance(content, np.ndarray):
				h5_file[f'{file_group}'].create_dataset(f'{file_group}/images', data=content, compression='gzip')

			return h5_file.keys()