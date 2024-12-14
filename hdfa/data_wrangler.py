import os
import logging
import h5py as h5
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from zipfile import ZipFile
from typing import AnyStr, Dict, List, Tuple
import numpy as np
from PIL import Image
import tifffile
import json
import csv
import random
import gzip


class _H5FileCreator:

	def __init__(self, output_file: AnyStr, write_mode: AnyStr) -> None:
		self.__output_file = output_file
		self.__write_mode = write_mode

	def _create_file(self) -> h5.File:
		h5_file = h5.File(self.__output_file, mode=self.__write_mode, libver='latest',
								   locking=True, driver='core', stdio=True, split=True, ndcc_bytes=65536)
		return h5_file


class H5DataCreator:
	def __init__(self, output_file: AnyStr, input_file: AnyStr, input_dict: Dict | np.ndarray = None) -> None:
		"""
        Initialize the H5DataCreator class.

        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        """
		self.__h5_file = self.__initialize_h5_file()
		self.__input_file = input_file
		self.__output_file = output_file
		self.__input_dict = input_dict
		# Initialize logger for this class
		self.__logger = logging.getLogger(__name__)
		self.__h5_file = self.__initialize_h5_file()

		# Optionally process input dictionary if provided
		if self.__input_dict:
			self.__logger.warning("Dictionary provided. It will be processed.")
			self.input_processor()

	def __initialize_h5_file(self) -> h5.File:
		"""
        Creates or appends to the HDF5 file depending on whether it exists.

        :return: An HDF5 file object.
        """
		if os.path.exists(self.__output_file):
			self.__logger.warning("Appending to existing file")
			write_mode = 'a'
		else:
			self.__logger.warning("Creating new file")
			write_mode = 'w'
		return _H5FileCreator(output_file=self.__output_file, write_mode=write_mode)._create_file()

	def random_int_generator(self) -> int:
		random_int = random.randint(1, 10000)
		return random_int

	def __classify_inputs(self, file_list: List[AnyStr], open_file: ZipFile | h5.File | gzip.GzipFile) -> Tuple[List, List]:
		"""Classify and process input files into structured data formats."""
		content_list: List = []
		processed_file_list: List[str] = []

		image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
		for file in file_list:
			try:
				# Process JSON files
				if file.endswith('.json'):
					raw_content = open_file.read(file).decode('utf-8').splitlines()
					content_list.extend(json.loads(row) for row in raw_content)
					processed_file_list.append(file)
				# Process image files (JPEG, PNG, BMP, TIFF)
				elif file.endswith(image_extensions):
					if file.endswith('.tiff'):
						img = tifffile.imread(file)
					else:
						with open_file.open(file) as img_file:
							img = Image.open(img_file)
							img = np.array(img)
					content_list.append(img)
					processed_file_list.append(file)
				# Process SVG files (Convert to numpy array via PNG generation)
				elif file.endswith('.svg'):
					temp_img = f"{self.random_int_generator()}_temp.png"
					try:
						drawing = svg2rlg(open_file.open(file))
						renderPM.drawToFile(drawing, temp_img, fmt='PNG')
						img = Image.open(temp_img)
						img_array = np.array(img)
						content_list.append(img_array)
						processed_file_list.append(file)
					finally:
						# Ensure temp file is removed, even in case of failures.
						if os.path.exists(temp_img):
							os.remove(temp_img)
				# Process CSV files
				elif file.endswith('.csv'):
					with open_file.open(file) as csv_file:
						csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",")
						content = [row for row in csv_reader]
						content_list.append(content)
						processed_file_list.append(file)

			except Exception as e:
				self.__logger.getLogger(__name__).error(f"Error processing file {file}: {e}")
				continue
		return content_list, processed_file_list

	def __parse_data(self, input_dict: Dict | np.ndarray) -> List:
		"""
	    Recursively parses a nested dictionary or a numpy array to extract and organize
	    data into a list of key-value pairs.

	    :param input_dict: Dictionary or numpy array to parse.
	    :return: List of key-value pairs or numpy arrays.
	    """
		value_list: List = []

		if isinstance(input_dict, Dict):
			for k, v in input_dict.items():
				if isinstance(v, dict):
					# Recursive call for nested dictionaries
					value_list.extend(self.__parse_data(v))
				elif isinstance(v, (list, np.ndarray)):
					# Check if the list or array contains integers
					if all(isinstance(i, int) for i in v):
						# Ensure v is converted to a numpy array only when needed
						value_list.append((k, np.array(v)))
					else:
						# Add raw lists if not integers
						value_list.append((k, v))
				else:
					# Add primitive types (e.g., strings, numbers)
					value_list.append((k, v))
		elif isinstance(input_dict, np.ndarray):
			# If the input is a numpy array, append it directly
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

	def __open_zip(self) -> Tuple[List, List]:
		zip = ZipFile(self.__input_file, 'r')
		file_list = self.__file_list()
		content, processed_file_list = self.__classify_inputs(file_list, zip)
		return content, processed_file_list

	def __open_h5(self) -> Tuple[List, List]:
		h5file = self.__input_file.h5.open().read().decode('utf-8')
		content, processed_file_list = self.__classify_inputs([h5file], h5file)
		return content, processed_file_list

	def __open_gzip(self) -> Tuple[List, List]:
		gzfile = gzip.open(self.__input_file, 'r', encoding='utf-8')
		file_list = self.__file_list()
		content, processed_file_list = self.__classify_inputs(file_list, gzfile)
		return content, processed_file_list

	def __input_file_type(self) -> str | Tuple[List, List]:
		if self.__input_file.__getattribute__(__name=self.__input_file).endswith('.zip' or 'z'):
			content_list, file_list = self.__open_zip()
			return content_list, file_list
		elif self.__input_file.__getattribute__(__name=self.__input_file).endswith('.h5' or 'hdf5'):
			content_list, file_list = self.__open_h5()
			return content_list, file_list
		elif self.__input_file.__getattribute__(__name=self.__input_file).endswith('.gz' or 'gzip'):
			content_list, file_list = self.__open_gzip()
			return content_list, file_list
		else:
			return 'unknown'

	def input_processor(self) -> Tuple[h5.File.keys, h5.File.filename]:
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
		self.__h5_file = h5.File(self.input_file, 'r', libver='latest', locking=True, driver='core', stdio=True)

	def recursive_retrieval(self, object_name: str):
		target = self.__h5_file.get(object_name)
		if isinstance(target, h5.Group):
			return {k: v[:] for k, v in target.items()}
		elif isinstance(target, h5.Dataset):
			return target[:]

	def __retrieve_all_data(self) -> Tuple[List, List]:
		group_data_list = [self.__h5_file[name][:] for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Group)]
		dataset_data_list = [self.__h5_file[name][:] for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Dataset)]
		return group_data_list, dataset_data_list

	def retrieve_groups_list(self) -> List:
		return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Group)]

	def retrieve_datasets_list(self) -> List:
		return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Dataset)]

	def retrieve_searched_group(self, searched_group: str) -> Tuple:
		group = self.__h5_file.get(searched_group)
		if isinstance(group, h5.Group):
			return searched_group, {k: v[:] for k, v in group.items()}
		elif isinstance(group, h5.Dataset):
			return searched_group, group[:]

	def retrieve_searched_dataset(self, searched_dataset: str) -> Tuple:
		dataset = self.__h5_file.get(searched_dataset)
		if isinstance(dataset, h5.Dataset):
			return searched_dataset, dataset[:]
		elif isinstance(dataset, h5.Group):
			return searched_dataset, {k: v[:] for k, v in dataset.items()}


