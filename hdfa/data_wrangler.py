import multiprocessing
import os
import logging

from setuptools.package_index import local_open

from mpLocal import mpQLocal
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


class H5FileCreator:

	def __init__(self, output_file: AnyStr, write_mode: AnyStr) -> None:
		self.__output_file = output_file
		self.__write_mode = write_mode

	def create_file(self) -> h5.File:
		__h5_file = h5.File(self.__output_file, mode=self.__write_mode, libver='latest', locking=True, driver='stdio')
		return __h5_file

class H5DataCreator:
	def __init__(self, output_file: AnyStr, input_file: AnyStr, input_dict: Dict | np.ndarray = None) -> None:
		"""
        Initialize the H5DataCreator class.

        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        """
		self.__input_file = input_file
		self.__output_file = output_file
		self.__input_dict = input_dict

		# Initialize logger for this class
		self.__logger = logging.getLogger(__name__)
		self.__h5_file = self.__initialize_h5_file

		# Optionally process input dictionary if provided
		if self.__input_dict:
			self.__logger.warning("Dictionary provided. It will be processed.")
			self.input_processor()

	@property
	def __initialize_h5_file(self) -> h5.File:
		"""
        Creates or appends to the HDF5 file depending on whether it exists.
        :return: An HDF5 file object.
        """
		return H5FileCreator(output_file=self.__output_file, write_mode='a').create_file()

	def random_int_generator(self) -> str:
		random_int = random.randint(1, 1000000)
		return str(random_int)

	def __write_content_to_file(self) -> None:
		self.__h5_file.flush()

	def __convert_images(self, local_q_process: multiprocessing.Process, file: AnyStr,
	                     open_file: ZipFile | h5.File | gzip.GzipFile, content_list: List, processed_file_list: List[str]):
		image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
		local_process = mpQLocal()
		local_process.join_mp_process(local_q_process)
		try:
			if file.endswith(image_extensions):
				ds = file.split('/')[0]
				if file.endswith('tiff'):
					img = tifffile.imread(file)
				else:
					with open_file.open(file) as img_file:
						img = np.array(Image.open(img_file))
				content_list.append([ds, img])
				processed_file_list.append(file)
				local_process.send_data(local_q_process, [content_list, processed_file_list])
			# Process SVG files (Convert to numpy array via PNG generation)
			elif file.endswith('svg'):
				ds = file.split('/')[0]
				temp_img = f"{self.random_int_generator()}_temp.png"
				try:
					drawing = svg2rlg(open_file.open(file))
					renderPM.drawToFile(drawing, temp_img, fmt='PNG')
					img = np.array(Image.open(temp_img))
					content_list.append([ds, img])
					processed_file_list.append(file)
					local_process.send_data(local_q_process, [content_list, processed_file_list])
				finally:
				# Ensure temp file is removed, even in case of failures.
					if os.path.exists(temp_img):
						os.remove(temp_img)
		except Exception as e:
			print(e)
			self.__logger.error(f"Error processing file {file}: {e}")

	def __classify_inputs(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile) -> Tuple:
		"""Classify and process input files content into structured data formats."""
		content_list: List = []
		processed_file_list: List[str] = []
		# Use multiprocessing and queues for large image lists
		local_process = mpQLocal()
		local_q_process, process_q = local_process.setup_mp_process()
		local_process.join_mp_process(local_q_process)

		image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
		try:
			# Process JSON files
			if file.endswith('json'):
				file_name = file.casefold()
				raw_content = open_file.read(file).decode('utf-8').splitlines()
				content = [row for row in raw_content]
				content = json.loads(content[0])
				content_list.append([file_name, content])
				line_count = len(content_list)
				processed_file_list.append(file_name + '-' + str(line_count))
				local_process.send_data(local_q_process,[content_list, processed_file_list])
			# Process image files
			elif file.endswith(image_extensions):
				self.__convert_images(local_q_process, file, open_file, content_list, processed_file_list)
			# Process CSV files
			elif file.endswith('csv'):
				with open_file.open(file) as csv_file:
					csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
					                        doublequote=True, quotechar='"')
					content = [row for row in csv_reader]
					content_list.append([file, content])
					processed_file_list.append(file)
					local_process.send_data(local_q_process, [content_list, processed_file_list])

		except Exception as e:
			self.__logger.getLogger(__name__).error(f"Error processing file {file}: {e}")
			print(e)

		return local_process, local_q_process

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
				if isinstance(v, Dict):
					# Recursive call for nested dictionaries
					value_list.extend(self.__parse_data(v))
				elif isinstance(v, (List, np.ndarray)):
					# Check if the list or array contains integers
					if all(isinstance(i, int) for i in v):
						# Ensure v is converted to a numpy array only when needed
						value_list.append((k, np.ndarray(v)))
					else:
						# Add raw lists if not integers
						value_list.append((k, v))
				elif isinstance(v, Tuple):
					_a, _b, _c = v
					value_list.append((k, [_a, _b, _c]))
				elif isinstance(v, (int, str)):
					# Add primitive types (e.g., strings, numbers)
					value_list.append((k, v))
		elif isinstance(input_dict, np.ndarray):
			# If the input is a numpy array, append it directly
			value_list.append(input_dict)
		elif isinstance(input_dict, Tuple):
			_a, _b, _c = input_dict
			value_list.append((_a, _b, _c))
		return value_list

	def __file_list(self) -> List:
		if self.__input_file.endswith('zip' or 'z'):
			with ZipFile(self.__input_file, 'r') as zip:
				file_list = zip.namelist()
				return file_list
		elif self.__input_file.endswith('gz' or 'gzip'):
			with gzip.open(self.__input_file, 'rb') as gzf:
				file_list = gzip.GzipFile(fileobj=gzf).fileobj.__dict__.get('namelist')
				return file_list

	def __open_zip(self) -> Tuple[List, List]:
		zip = ZipFile(self.__input_file, 'r')
		file_list = self.__file_list()
		for file in file_list:
			process, local_q = self.__classify_inputs(file, zip)
			return process, local_q

	def __open_h5(self) -> Tuple[List, List]:
		h5file = self.__input_file.h5.open().read().decode('utf-8')
		process, local_q = self.__classify_inputs(h5file, h5file)
		return process, local_q

	def __open_gzip(self) -> Tuple[List, List]:
		gzfile = gzip.open(self.__input_file, 'r', encoding='utf-8')
		file_list = self.__file_list()
		for file in file_list:
			process, local_q = self.__classify_inputs(file, gzfile)
			return process, local_q

	def __input_file_type(self) -> str | Tuple[List, List]:
		if self.__input_file.endswith('zip' or 'z'):
			process, local_q = self.__open_zip()
			return process, local_q
		elif self.__input_file.endswith('h5' or 'hdf5'):
			process, local_q = self.__open_h5()
			return process, local_q
		elif self.__input_file.endswith('gz' or 'gzip'):
			process, local_q = self.__open_gzip()
			return process, local_q
		else:
			return ['unknown'], ['unknown']

	@property
	def input_processor(self) -> h5.File.keys:
		try:
			if self.__input_file_type() != 'unknown':
				process, local_q = self.__input_file_type()
			elif self.__input_dict:
				content_list = [self.__input_dict]
				file_list: List = []

			if process and local_q:
				for data in process.recv_data(local_q):
					content_list, file_list = data
					for contents, files in content_list, file_list:
						for file_group, content_lines in contents:
							if not self.__h5_file.get(file_group):
								self.__h5_file.create_group(file_group, track_order=True)
								self.__h5_file[file_group].attrs['file_name'] = file_group
								self.__h5_file[file_group].attrs['content_list_length'] = len(content_list)
								print("write file group attrs")
								self.__write_content_to_file()

							elif not file_list:
								file_group = 'root'
								if not self.__h5_file.get(file_group):
									self.__h5_file.require_group(file_group)
									self.__h5_file[file_group].attrs['file_name'] = file_group
									self.__h5_file[file_group].attrs['content_list_length'] = len(content_list)
									print("write root attrs")
									self.__write_content_to_file()

					for file, line in content_list:
						if isinstance(line, Dict):
							list_count = 0
							kv_list = self.__parse_data(input_dict=line)
							kva = np.array(kv_list, dtype=np.dtype('S'))
							print(kva)
							idx = list_count
							self.__h5_file.require_group(file_group).attrs[f'{idx}'] = kva
							self.__write_content_to_file()
							print("write attrs")
							list_count += 1

						elif isinstance(line, np.ndarray):
							if self.__h5_file.get(f'{file_group}'):
								self.__h5_file[f'{file_group}'].create_dataset(f'{file_group}', data=line,
								                                               compression='gzip',
								                                               chunks=True)
								print("write ndarray")
								self.__write_content_to_file()
						else:
							self.__write_content_to_file()

			elif content_list and file_list:
				for contents, files in content_list:
					for file_group, content_lines in contents:
						if not self.__h5_file.get(file_group):
							self.__h5_file.create_group(file_group, track_order=True)
							self.__h5_file[file_group].attrs['file_name'] = file_group
							self.__h5_file[file_group].attrs['content_list_length'] = len(content_list)
							print("write file group attrs")

		except Exception as e:
			self.__h5_file.flush()
			self.__h5_file.close()
			self.__logger.error(f"Error processing input file: {e}")
			return e
		finally:
			self.__h5_file.close()
			return self.__h5_file.keys()


class H5DataRetriever:
	def __init__(self, input_file: str, group_list: List, dataset_list: List) -> None:
		self.__input_file = input_file
		self.__group_data_list = group_list
		self.__dataset_data_list = dataset_list
		self.__h5_file = h5.File(self.__input_file, 'r', libver='latest', locking=True, driver='None')

	def recursive_retrieval(self) -> Dict | h5.Dataset | None:
		if self.__group_data_list:
			target = [g for g in self.__group_data_list if g in self.__h5_file]
		elif self.__dataset_data_list:
			target = [d for d in self.__dataset_data_list if d in self.__h5_file]

			if isinstance(target, h5.Group):
				return {k: v for k, v in target.items()}
			elif isinstance(target, h5.Dataset):
				return {k: v for k, v in target.collective.__dict__()}
			else:
				return {k: v for k, v in h5.Group(target).attrs.items()}

	def retrieve_all_data_lists(self) -> Tuple[List, List]:
		group_data_list = [self.__h5_file[name] for name in self.__h5_file if
		                   isinstance(self.__h5_file[name], h5.Group)]
		dataset_data_list = [self.__h5_file[name] for name in self.__h5_file if
		                     isinstance(self.__h5_file[name], h5.Dataset)]
		return group_data_list, dataset_data_list

	def retrieve_group_attrs_data(self) -> List[Dict]:
		attrs_list: List = []
		groups = self.retrieve_group_list()
		for g in groups:
			attrs_list.append(self.__h5_file[g].attrs.items())
		return attrs_list

	def retrieve_group_list(self) -> List:
		return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Group)]

	def retrieve_dataset_list(self) -> List:
		return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Dataset)]

	def retrieve_searched_group(self, searched_group: str) -> Tuple:
		group = self.__h5_file.get(searched_group)
		if isinstance(group, h5.Group):
			return (searched_group, {k: v for k, v in group.items()}, [i for i in group.attrs.keys() if i])

	def retrieve_searched_dataset(self, searched_dataset: str) -> Tuple:
		dataset = self.__h5_file.get(searched_dataset)
		if isinstance(dataset, h5.Dataset):
			return (searched_dataset, {item for item in dataset.chunks}, [i for i in dataset.attrs.keys() if i])
