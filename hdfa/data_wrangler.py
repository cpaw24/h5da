import h5py as h5
from zipfile import ZipFile
from typing import AnyStr, Dict, List
import numpy as np
from PIL import Image
import json
import csv


class H5DataFileCreator:
	def __init__(self, output_file: AnyStr) -> None:
		self.output_file = output_file
		self.h5_file = h5.File(self.output_file, 'w', libver='latest',
		                       meta_block_size=8388608, locking=True, driver='core')

class H5DataExtractor:
	def __init__(self, output_file: AnyStr,  input_file: AnyStr, input_dict: Dict | np.ndarray) -> None:
		self.input_file = input_file
		self.input_dict = input_dict
		self.output_file = output_file
		self.h5_file = H5DataFileCreator(output_file=self.output_file)

	def __parse_data(self, input_dict: Dict | np.ndarray) -> List:
		value_list: List = []
		if isinstance(input_dict, Dict):
			for k, v in input_dict.items():
				if isinstance(v, dict):
					self.__parse_data(v)
				elif isinstance(v, list):
					v = np.array(v)
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

	def __file_list(self, input_file: AnyStr) -> List:
		with ZipFile(input_file) as zip:
			file_list = zip.namelist()
			return file_list

	def __open_zip(self, input_file: AnyStr) -> List:
		with ZipFile.open(name=input_file, mode='r') as zip:
			if input_file.endswith('.json'):
				content = zip.read().decode('utf-8').rstrip().splitlines()
			elif input_file.endswith('.jpg' or '.jpeg' or '.png' or '.bmp' or '.tiff' or '.svg'):
				img = Image.open(zip)
				data = np.array(img)
				content = self.__parse_data(data)

				return content

	def __dataset_content(self, input_file: AnyStr) -> tuple[List, List]:
		processed_file_list: List = []
		file_list = self.__file_list(input_file)
		for file in file_list:
			processed_file_list.append(file)
			contents = self.__open_zip(file)

			if file.endswith('.csv'):
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
					elif isinstance(v, str | np.array):
						h5_file[f'{file_group}'].attrs[k] = v
			elif isinstance(content, np.ndarray):
				h5_file[f'{file_group}'].create_dataset(f'{file_group}/images', data=content, compression='gzip')

			return h5_file.keys(), h5_file.filename

