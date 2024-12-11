import os
import h5py as h5
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from zipfile import ZipFile
from typing import AnyStr, Dict, List
import numpy as np
from PIL import Image
import tifffile
import json
import csv
import random


class H5FileCreator:

	def __init__(self, output_file: AnyStr, write_mode: AnyStr) -> None:
		self.output_file = output_file
		self.write_mode = write_mode

	def create_file(self) -> h5.File:
		h5_file = h5.File(self.output_file, mode=self.write_mode, libver='latest',
		                       meta_block_size=8388608, locking=True, driver='core')
		return h5_file


class H5DataCreator:
	def __init__(self, output_file: AnyStr, input_file: AnyStr, input_dict: Dict | np.ndarray) -> None:
		self.__input_file = input_file
		self.__input_dict = input_dict
		self.__output_file = output_file
		self.__h5_file = H5FileCreator(output_file=self.__output_file, write_mode='w').create_file()

	def random_int_generator(self) -> int:
		random_int = random.randint(1, 10000)
		return random_int

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
		with ZipFile(input_file, 'r') as zip:
			file_list = zip.namelist()
			return file_list

	def __open_zip(self, input_file: AnyStr) -> tuple[List, List]:
		with ZipFile(input_file, 'r') as zip:
			content_list: List = []
			file_list = self.__file_list(input_file)
			processed_file_list: List[str] = []
			for file in file_list:
				if file.endswith('.json'):
					raw_content = zip.read(file).decode('utf-8').rstrip().splitlines()
					for row in raw_content:
						row_dict = json.loads(row)
						content_list.append(row_dict)
					processed_file_list.append(file)

				elif file.endswith('.jpg' or '.jpeg' or '.png' or '.bmp' or '.tiff'):
					if file.endswith('.tiff'):
						img = tifffile.imread(file)
						content_list.append(img)
					else:
						img = Image.open(file)
						data = np.array(img)
						content_list.append(data)
					processed_file_list.append(file)

				elif file.endswith('.svg'):
					temp_img = f"{self.random_int_generator()}_temp.png"
					drawing = svg2rlg(file)
					renderPM.drawToFile(drawing, f"{temp_img}", fmt='png')
					img = Image.open(f'{temp_img}')
					img_array = np.array(img)
					content_list.append(img_array)
					processed_file_list.append(file)
					os.remove(temp_img)

				elif file.endswith('.csv'):
					with csv.reader(file, 'r', delimiter=",", quotechar="") as datafile:
						content = datafile.read().decode('utf-8')
						content_list.append(content)
						processed_file_list.append(file)

			return content_list, processed_file_list

	def input_processor(self) -> tuple[h5.File.keys, h5.File.filename]:
		content_list, file_list = self.__open_zip(self.__input_file)
		for line, file_nm in zip(content_list, file_list):
			file_group = file_nm.split('.')[0]
			self.__h5_file.create_group(file_group)
			if isinstance(line, Dict):
				kv_list = self.__parse_data(input_dict=line)
				for kv in kv_list:
					k, v = kv
					if isinstance(v, np.ndarray):
						self.__h5_file[f'{file_group}'].create_dataset(k, data=v, compression='gzip')
					elif isinstance(v, str or np.array):
						self.__h5_file[f'{file_group}'].attrs[k] = v
			elif isinstance(line, np.ndarray):
				self.__h5_file[f'{file_group}'].create_dataset(f'{file_group}/images', data=line, compression='gzip')
			else:
				self.__h5_file.file.close()

		return self.__h5_file.keys(), self.__h5_file.filename


class H5DataRetriever:
	def __init__(self, input_file: h5.File, group_list: List,  dataset_list: List) -> None:
		self.__input_file = input_file
		self.__group_list = group_list
		self.__dataset_list = dataset_list
		self.__h5_file = h5.File(self.__input_file, 'r', libver='latest', locking=True, driver='core')

	def retrieve_data(self) -> tuple[List, List]:
		file = self.__h5_file
		group_data_list: List = []
		dataset_data_list: List = []

		for __group in self.__group_list:
			__group = file.require_group(__group)
			__group_data = file.get(__group)
			group_data_list.append(__group_data)

		for __dataset in self.__dataset_list:
			if group_data_list:
				for group in group_data_list:
					__dataset_group = file.require_group(group)
			else:
				__dataset_group = file.require_group(__dataset)

				if __dataset_group:
					__dataset_data = __dataset_group.get(__dataset)
					dataset_data_list.append(__dataset_data)
				else:
					__dataset_data = file.get(__dataset)
					dataset_data_list.append(__dataset_data)

			return group_data_list, dataset_data_list

