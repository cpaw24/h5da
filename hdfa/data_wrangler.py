import os
import logging
import h5py as h5
from h5py._hl import dataset
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
import gzip


class H5FileCreator:

	def __init__(self, output_file: AnyStr, write_mode: AnyStr) -> None:
		self.__output_file = output_file
		self.__write_mode = write_mode

	def create_file(self) -> h5.File:
		h5_file = h5.File(self.__output_file, mode=self.__write_mode, libver='latest',
		                       locking=True, driver='core', stdio=True, split=True, ndcc_bytes=65536)
		return h5_file


class H5DataCreator:
	def __init__(self, output_file: AnyStr, input_file: AnyStr, input_dict: Dict | np.ndarray) -> None:
		self.__input_file = input_file
		self.__input_dict = input_dict
		self.__output_file = output_file
		self.__logger = logging
		if os.path.exists(self.__output_file):
			self.__logger.getLogger(__name__).warning("Appending to existing file")
			self.__h5_file = H5FileCreator(output_file=self.__output_file, write_mode='a').create_file()
		else:
			self.__logger.getLogger(__name__).warning("Creating new file")
			self.__h5_file = H5FileCreator(output_file=self.__output_file, write_mode='w').create_file()
		if self.__input_dict:
			self.__logger.getLogger(__name__).warning("Sending dictionary to input processor")
			self.input_processor()

	def random_int_generator(self) -> int:
		random_int = random.randint(1, 10000)
		return random_int

	def __classify_inputs(self, file_list: List[AnyStr], open_file: ZipFile | h5.File | gzip.GzipFile) -> tuple[List, List]:
		content_list: List = []
		processed_file_list: List[str] = []
		for file in file_list:
			# JSON to attributes
			if file.endswith('.json'):
				raw_content = open_file.read(file).decode('utf-8').rstrip().splitlines()
				for row in raw_content:
					row_dict = json.loads(row)
					content_list.append(row_dict)
				processed_file_list.append(file)
			# JPEG, PNG, TIFF, BMP to numpy array
			elif file.endswith('.jpg' or '.jpeg' or '.png' or '.bmp' or '.tiff'):
				if file.endswith('.tiff'):
					img = tifffile.imread(file)
					content_list.append(img)
				else:
					img = Image.open(file)
					data = np.array(img)
					content_list.append(data)
				processed_file_list.append(file)
			# SVG to numpy array
			elif file.endswith('.svg'):
				temp_img = f"{self.random_int_generator()}_temp.png"
				drawing = svg2rlg(file)
				renderPM.drawToFile(drawing, f"{temp_img}", fmt='png')
				img = Image.open(f'{temp_img}')
				img_array = np.array(img)
				content_list.append(img_array)
				processed_file_list.append(file)
				os.remove(temp_img)
			# CSV
			elif file.endswith('.csv'):
				with csv.reader(file, 'r', delimiter=",", quotechar="") as datafile:
					content = datafile.read().decode('utf-8')
					content_list.append(content)
					processed_file_list.append(file)
			return content_list, processed_file_list

	def __parse_data(self, input_dict: Dict | np.ndarray) -> List:
		value_list: List = []
		if isinstance(input_dict, Dict):
			for k, v in input_dict.items():
				if isinstance(v, dict):
					self.__parse_data(v)
				elif isinstance(v, list) or isinstance(v, np.ndarray):
					int_array = [i for i in v if i == int]
					if int_array:
						v = np.ndarray(v)
						value_list.append([k, v])
						continue
					elif isinstance(v, list) and not int_array:
						value_list.append([k, v])
						continue
		elif isinstance(input_dict, np.ndarray):
			value_list.append(input_dict)

		return value_list

	def __file_list(self) -> List:
		if self.__input_file.endswith('.zip' or '.z'):
			with ZipFile(self.__input_file, 'r') as  zip:
				file_list = zip.namelist()
				return file_list
		elif self.__input_file.endswith('.gz' or 'gzip'):
			with gzip.open(self.__input_file, 'rb') as gzf:
				file_list = gzip.GzipFile(fileobj=gzf).fileobj.__dict__.get('namelist')
				return file_list

	def __open_zip(self) -> tuple[List, List]:
		zip = ZipFile(self.__input_file, 'r')
		file_list = self.__file_list()
		content, processed_file_list = self.__classify_inputs(file_list, zip)
		return content, processed_file_list

	def __open_h5(self) -> tuple[List, List]:
		h5file = self.__input_file.h5.open().read().decode('utf-8')
		content, processed_file_list = self.__classify_inputs([h5file], h5file)
		return content, processed_file_list

	def __open_gzip(self) -> tuple[List, List]:
		gzfile = gzip.open(self.__input_file, 'r', encoding='utf-8')
		file_list = self.__file_list()
		content, processed_file_list = self.__classify_inputs(file_list, gzfile)
		return content, processed_file_list

	def __input_file_type(self) -> str | tuple[List, List]:
		if self.__input_file.__getattribute__(__name=self.__input_file).endswith('.zip'):
			content_list, file_list = self.__open_zip()
			return content_list, file_list
		elif self.__input_file.__getattribute__(__name=self.__input_file).endswith('.h5'):
			content_list, file_list = self.__open_h5()
			return content_list, file_list
		elif self.__input_file.__getattribute__(__name=self.__input_file).endswith('.gz'):
			content_list, file_list = self.__open_gzip()
			return content_list, file_list
		else:
			return 'unknown'

	def input_processor(self) -> tuple[h5.File.keys, h5.File.filename]:
		if self.__input_file_type() != 'unknown':
			content_list, file_list = self.__input_file_type()
		elif self.__input_dict:
			content_list = [self.__input_dict]
			file_list: List = []

			for line, file_nm in zip(content_list, file_list):
				if file_list:
					file_group = file_nm.split('.')[0]
					self.__h5_file.create_group(file_group, track_order=True)
				else:
					file_group = 'root'
				if isinstance(line, Dict):
					kv_list = self.__parse_data(input_dict=line)
					for kv in kv_list:
						k, v = kv
						if isinstance(v, np.ndarray):
							self.__h5_file[f'{file_group}'].create_dataset(k, data=v,
							                                               compression='gzip', chunks=True)
						elif isinstance(v, str or np.array):
							self.__h5_file[f'{file_group}'].attrs[k] = v
				elif isinstance(line, np.ndarray):
					self.__h5_file[f'{file_group}'].create_dataset(f'{file_group}/images',
					                                               data=line, compression='gzip', chunks=True)
				else:
					self.__h5_file.file.close()
		return self.__h5_file.keys(), self.__h5_file.filename


class H5DataRetriever:
	def __init__(self, input_file: h5.File, group_list: List,  dataset_list: List) -> None:
		self.input_file = input_file
		self.group_data_list = group_list
		self.dataset_data_list = dataset_list
		self.__h5_file = h5.File(self.input_file, 'r', libver='latest', locking=True, driver='core')

	def recursive_retrieval(self, object_name: str):
		for group in self.__h5_file.keys():
			if isinstance(group, h5.Group) and group.name == object_name:
				group_data = group.require_group(group.name)
				group_data = group_data.get(group_data)
				return group_data
			elif isinstance(group, h5.Dataset) and group.name == object_name:
				for ds_chunk in group.iter_chunks():
					if isinstance(ds_chunk, h5.Dataset):
						dataset_group = group.__getattribute__(group.name).require_group()
						for dataset_chunk in dataset.Dataset.iter_chunks():
							dataset_chunks = dataset_group.get(dataset_chunk)
							yield dataset_chunks.accumulate()
			else:
				self.recursive_retrieval(object_name)

	def __retrieve_all_data(self) -> tuple[List, List]:
		group_data_list: List = []
		for group in self.__h5_file.keys():
			if isinstance(group, h5.Group):
				group = group.require_group(group.name)
				group_data = group.get(group)
				group_data_list.append(group_data)
		for dataset in self.__h5_file.keys():
			if isinstance(dataset, h5.Dataset):
				for ds_chunk in dataset.iter_chunks():
					if isinstance(ds_chunk, h5.Dataset):
						dataset_group = dataset.__getattribute__(dataset.name).require_group()
						dataset_data = dataset_group.get(ds_chunk)
						yield group_data_list, dataset_data.accumulate()

	def __retrieve_groups_list(self) -> List[h5.Group]:
		group_data_list: List = []
		for group in self.__h5_file.keys():
			if isinstance(group, h5.Group):
				group_data_list.append(group)
				for item in group.items():
					if isinstance(item, h5.Dataset):
						group_data_list.append(f" ds:{item.name}")
					elif isinstance(item, h5.Group):
						group_data_list.append(item)

		return group_data_list

	def __retrieve_datasets_list(self) -> List[h5.Dataset]:
		dataset_data_list: List = []
		for dataset in self.__h5_file.keys():
			if isinstance(dataset, h5.Dataset):
				dataset_data_list.append(dataset.name)
		return dataset_data_list

	def __retrieve_searched_group(self, searched_group: str) -> tuple[h5.Group.name, h5.Group | h5.Dataset]:
		for group in self.__h5_file.keys():
			if isinstance(group, h5.Group):
				if group.name == searched_group:
					group_data = group.require_group(group.name)
					group_data = group_data.get(group_data)
					yield group.name, group_data
			elif isinstance(group, h5.Dataset):
				if group.name == searched_group:
					self.recursive_retrieval(group.name)

	def __retrieve_searched_dataset(self, searched_dataset: str) -> tuple[h5.Dataset.name, h5.Dataset | h5.Group]:
		for dataset in self.__h5_file.keys():
			if isinstance(dataset, h5.Dataset):
				if dataset.name == searched_dataset:
					dataset_group = h5.Group.require_group(self.__h5_file, dataset.name)
					dataset_data = dataset_group.get(dataset_group)
					for dataset_chunk in dataset_data.iter_chunks():
						if isinstance(dataset_chunk, h5.Dataset):
							dataset_data = dataset_data.accumulate()
							yield dataset.name, dataset_data
			elif isinstance(dataset, h5.Group):
				if dataset.name == searched_dataset:
					self.recursive_retrieval(dataset.name)


