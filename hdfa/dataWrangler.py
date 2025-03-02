import gc
import io
import math

from distributed.utils_test import double

from imageProcessor import ImageProcessor
from schemaProcessor import SchemaProcessor
from videoProcessor import VideoProcessor
from textFileProcessor import TextFileProcessor
import multiprocessing
import os
import uuid
import tarfile
import zipfile
from datetime import datetime

from hdfa.mpLocal import MpQLocal
import h5py as h5
from zipfile import ZipFile
from typing import AnyStr, Dict, List, Tuple
import numpy as np
import json
import random
import gzip
import h5rdmtoolbox as h5tbx


class DataProcessor:
    def __init__(self, output_file: AnyStr, input_file: AnyStr,
                 input_dict: Dict | np.ndarray = None, write_mode: AnyStr = 'a',
                 schema_file: AnyStr = None, config_file: AnyStr = None) -> None:
        """
        Initialize the H5DataCreator class.
        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz, .h5, .tar, .tar.gz).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        :param write_mode: Optional mode to write the HDF5 file. Default is 'a' for append mode.
        :param schema_file: Optional schema file.
        :param config_file: Optional configuration file.
        """
        self.__input_file = input_file
        self.__input_dict = input_dict
        self.__write_mode = write_mode
        self.__schema_file = schema_file
        self.__config_file = config_file
        self.__output_file = output_file
        self.__initialize()

    def __initialize(self):
        self.__h5_file = self.__create_file(self.__output_file)
        self.__schema_dict = json.load(open(self.__schema_file, 'r'))
        self.__config_dict = json.load(open(self.__config_file, 'r'))
        """ Initialize image, video, schema, and text processors """
        self._img_processor = ImageProcessor(self.__output_file,
                                             schema_dict=self.__schema_dict,
                                             config_dict=self.__config_dict)
        self._vid_processor = VideoProcessor()
        self._txt_processor = TextFileProcessor()
        self._schema_processor = SchemaProcessor(output_file=self.__output_file,
                                                 schema_file=self.__schema_file,
                                                 config_file=self.__config_file
        )

    @staticmethod
    def random_int_generator() -> str:
        random_int = random.randint(1, 1000000)
        return str(random_int)

    @classmethod
    def __create_file(cls, __output_file: AnyStr) -> h5.File:
        """Create an HDF5 file. Using customized page size and buffer size to reduce memory usage."""
        """ Set the kwargs values for the HDF5 file creation."""
        kwarg_vals = {'libver': 'latest', 'driver': 'core', 'backing_store': True, 'fs_persist': True,
                      'fs_strategy': 'page', 'fs_page_size': 65536, 'page_buf_size': 655360}

        with h5tbx.File(__output_file, mode='w', **kwarg_vals) as __h5_file:
            return __h5_file

    @staticmethod
    def _flush_content_to_file(h5_file: h5.File) -> None:
        """Flush content from memory to HDF5 file."""
        h5_file.flush()

    @staticmethod
    def _gen_dataset_name():
        return 'ds-' + str(uuid.uuid4())[:8]

    @staticmethod
    def update_file_group(h5_file: h5.File, file_group: AnyStr, attr: int | AnyStr, attr_value: AnyStr | int) -> None:
        """Update a file group attribute with input values.
        :param h5_file: HDF5 file.
        :param file_group: HDF5 file group.
        :param attr: HDF5 group attribute in either integer or string format.
        :param attr_value: HDF5 group attribute value in either integer or string format.
        :return: None"""
        h5_file.require_group(file_group).attrs[f'{attr}'] = attr_value
        return h5_file.flush()

    def find_list_depth(self, obj: List | AnyStr) -> int | None:
        """Find the maximum depth of a nested list, and return the value.
        :param obj: List or string."""
        if not isinstance(obj, list):
            return 0
        if not obj:
            return 1
        return 1 + max(self.find_list_depth(item) for item in obj)

    def set_dataset_attributes(self, h5_file: h5.File, file_group: AnyStr, file: AnyStr) -> None:
        """Standard set of attributes to write per dataset.
        :param h5_file: HDF5 file object.
        :param file_group: HDF5 group.
        :param file: input file from archive.
        :return: None"""
        try:
            if isinstance(file, str):
                file = file
            elif isinstance(file, bytes):
                file = file.decode('utf-8')
            elif isinstance(file, io.BytesIO):
                file = file.read().decode('utf-8')
            elif isinstance(file, int):
                file = str(file)

            self.update_file_group(h5_file=h5_file, file_group=file_group,
                                   attr='source-file-name', attr_value=file.split('/')[-1])
            self.update_file_group(h5_file=h5_file, file_group=file_group,
                                   attr='file-type-extension', attr_value=file.split('.')[-1])
            self.update_file_group(h5_file=h5_file, file_group=file_group,
                                   attr='source-root', attr_value=file.split('/')[0])
        except AttributeError as ae:
            print(f'AttributeError: {ae, ae.args}')
        except Exception as e:
            print(f'Attribute Exception: {e, e.args}')

    def create_dataset_from_dict(self, h5_file: h5.File, name: AnyStr, data: Dict, file_group: AnyStr) -> None:
        """Create a dataset from a dictionary. Numpy arrays are string-ified and converted to byte strings.
        :param h5_file: HDF5 file object.
        :param name: HDF5 dataset name.
        :param data: HDF5 dataset data in dictionary format.
        :param file_group: HDF5 group.
        :return: None"""
        try:
            kv_list = self.parse_data(input_dict=data)
            kv_dict = kv_list[0]
            kvl = {str.encode(k): str.encode(v) for (k, v) in kv_dict.items()}
            kva = np.array(kvl, dtype=np.dtype('S'))
            h5_file.require_group(file_group).create_dataset(name, data=kva, compression='gzip')
        except ValueError as ve:
            print(f'Dataset from Dict ValueError: {ve, ve.args}')
        except TypeError as te:
            print(f'Dataset from Dict TypeError: {te, te.args}')
        except Exception as e:
            print(f'Dataset from Dict Exception: {e, e.args}')

    def create_dataset_from_input(self, h5_file: h5.File, data: List | Dict | np.ndarray, file_group: AnyStr,
                                  file: AnyStr) -> None:
        """Create a dataset from a list, dictionary, or Numpy array. Applying updates to group attributes.
        :param h5_file: HDF5 file object.
        :param data: HDF5 dataset data in list, dictionary, or numpy array.
        :param file_group: HDF5 group.
        :param file: file name from input source.
        :return: None"""
        try:
            gc.collect()
            h5_file.flush()
            if isinstance(data, List):
               if len(data) == 1 and isinstance(data[0], Dict):
                  name = self._gen_dataset_name()
                  self.create_dataset_from_dict(h5_file=h5_file, name=name, data=data[0], file_group=file_group)
                  self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)
                  print(f"write dataset {name} - {file}")

               elif len(data) == 1 and isinstance(data[0], List):
                   obj_depth = self.find_list_depth(data[0])
                   for i in range(obj_depth):
                      while i <= obj_depth:
                       if isinstance(data[0][i], List):
                            for dat in data[0][i]:
                               name = self._gen_dataset_name()
                               obj = str(dat)
                               kva = np.array(obj.encode('utf-8'), dtype=np.dtype('S'))
                               h5_file.require_group(file_group).create_dataset(name=name, data=kva, compression='gzip')
                               self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)
                               print(f"write dataset {name} - {file}")
                       elif isinstance(data[0][i], str):
                          obj2 = data[0][i]
                          self.update_file_group(h5_file=h5_file, file_group=file_group,
                                                 attr='dataset-file', attr_value=obj2)

               elif len(data) == 1 and isinstance(data[0], str | int | float | bool | datetime):
                  obj = str(data[0])
                  self.update_file_group(h5_file=h5_file, file_group=file_group,
                                         attr='file-group-attribute', attr_value=obj)

            elif isinstance(data, Dict):
               name = self._gen_dataset_name()
               self.create_dataset_from_dict(h5_file=h5_file, name=name, data=data, file_group=file_group)
               print(f"write dataset {name} - {file}")
               self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

            elif isinstance(data, np.ndarray):
               name = self._gen_dataset_name()
               h5_file.require_group(file_group).create_dataset(name=name, data=data, compression='gzip')
               print(f"write dataset {name} - {file}")
               self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

            gc.collect()
            return h5_file.flush()
        except ValueError as ve:
            print(f'Dataset from Input ValueError: {ve, ve.args}')
        except TypeError as te:
            print(f'Dataset from input TypeError: {te, te.args}')
        except Exception as e:
            print(f'Dataset from Input Exception: {e, e.args}')

    def create_file_group(self, h5_file: h5.File, group_name: AnyStr, content_size: int = 0):
        """Create a new file group in the HDF5 file.
        :param h5_file: HDF5 file object.
        :param group_name: New file group name.
        :param content_size: New file group content size.
        :return None"""
        created = self._get_file_group(h5_file, group_name)
        if not created:
            h5_file.create_group(group_name, track_order=True)
            h5_file[group_name].attrs['file_name'] = h5_file.name
            h5_file[group_name].attrs['schema_file'] = self.__schema_file
            h5_file[group_name].attrs['content_size'] = content_size
            print(f"write group {group_name} attrs")
            return h5_file.clear()

    @staticmethod
    def _get_file_group(h5_file: h5.File, group_name: AnyStr):
        return h5_file.get(group_name)

    def classify_inputs(self, file: AnyStr | io.BytesIO,
                        open_file: ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile | io.BytesIO,
                        process_q: multiprocessing.Queue) -> None:
        """Classify and process input file content into structured data formats. Valid inputs types are JSON, CSV,
        video(MP4/MP3), and image files including PNG, JPEG, BMP, TIFF, SVG, BMP, GIF, and ICO. Add them to multiprocessing.Queue.
        :param file: Path to the input file in archive (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param process_q: multiprocessing.Queue"""
        content_list: List = []
        processed_file_list: List[str] = []
        image_extensions = self.__config_dict.get('image_extensions')
        video_extensions = self.__config_dict.get('video_extensions')
        try:
            if not file.endswith('/'):
                """ Process JSON files """
                if file.endswith('json') or file.endswith('jsonl'):
                    content_list, processed_file_list = self._txt_processor.process_json(file, open_file,
                                                                          content_list, processed_file_list)
                    if content_list and processed_file_list:
                       process_q.put([content_list, processed_file_list, file])
                    """ Process image files """
                elif file.split('.')[-1] in image_extensions:
                    content_list, processed_file_list = self._img_processor.convert_images(file, open_file,
                                                                            content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])
                    """ Process CSV files """
                elif file.endswith('csv'):
                    content_list, processed_file_list = self._txt_processor.process_csv(file, open_file,
                                                                         content_list, processed_file_list)
                    if content_list and processed_file_list:
                       process_q.put([content_list, processed_file_list, file])
                    """ Process video files """
                elif file.split('.')[-1] in video_extensions:
                    content_list, processed_file_list = self._vid_processor.process_video(file, open_file,
                                                                           content_list, processed_file_list)
                    if content_list and processed_file_list:
                       process_q.put([content_list, processed_file_list, file])

        except Exception as e:
            print(f'classify_inputs Exception: {e, e.args}')
            pass

    def parse_data(self, input_dict: Dict | np.ndarray) -> List:
        """
        Recursively parses a nested dictionary or a numpy array to extract and organize
        data into a list of key-value pairs.
        :param input_dict: Dictionary or numpy array to parse.
        :return: List of --> key-value pairs, numpy arrays, or possibly tuples.
        """
        value_list: List = []
        if isinstance(input_dict, Dict):
            for k, v in input_dict.items():
                if isinstance(v, Dict):
                    """ Recursive call for nested dictionaries """
                    value_list.extend([self.parse_data(v)])
                elif isinstance(v, (List, np.ndarray)):
                    """ Check if the list or array contains integers """
                    if all(isinstance(i, int) for i in v):
                        """ Ensure v is converted to a numpy array only when needed """
                        value_list.append([k, np.ndarray(v)])
                    else:
                        """ Add raw lists if not integers """
                        value_list.append([k, v])
                elif isinstance(v, Tuple):
                    _a, _b, _c = v
                    value_list.append([k, [_a, _b, _c]])
                elif isinstance(v, (int, str)):
                    """ Add primitive types (e.g., strings, numbers) """
                    value_list.append([k, v])
        return value_list

    def process_content_list(self, content: List, file_group: AnyStr, h5_file: h5.File = None):
        try:
            for file, line in content:
                self.create_dataset_from_input(h5_file=h5_file, data=line, file_group=file_group, file=file)
                self._flush_content_to_file(h5_file=h5_file)
        except Exception as e:
            print(f'process_content_list Exception: {e}')
        finally:
            h5_file.flush()

    def start_mp(self) -> [multiprocessing.Queue, multiprocessing.Process]:
        """ Use multiprocessing and queues for large number of images; requires batch limits to be set in order to control
         the number of items in the queue and free memory on the host.
        :return: process_q which is a multiprocessing queue, and local_mpq which is a multiprocessing process"""
        try:
            multiprocessing.process.allow_run_as_child = True
            multiprocessing.process.allow_connection_pickling = True
            multiprocessing.process.inherit_connections = True
            multiprocessing.process.allow_multiprocess_child = True
            multiprocessing.set_start_method('fork', force=True)
            local_mpq = MpQLocal()
            process_q = local_mpq.get_queue()
            return process_q, local_mpq
        except Exception as e:
            print(f'start_mp Exception: {e, e.args}')

    def file_list(self, open_file: zipfile.ZipFile | tarfile.TarFile | gzip.GzipFile | io.BytesIO) -> None | List:
        """Get a list of files from ZIP or GZIP files."""
        if isinstance(open_file, zipfile.ZipFile):
            valid = zipfile.is_zipfile(open_file)
            if valid:
               file_list = open_file.namelist()
               return file_list
        elif isinstance(open_file, tarfile.TarFile):
                file_list = open_file.getnames()
                return file_list
        elif self.__input_file.endswith('gz' or 'gzip'):
            with gzip.open(self.__input_file, 'rb') as gz:
                buff = gz.read()
                file_list = gzip.decompress(buff).decode('utf-8')
                return file_list
        else:
            return []

    @staticmethod
    def __size_batching(file_list: List, batch_size: int):
        """Compute and slice Batch file list into a specified size for processing.
        :param file_list: List of files to process.
        :param batch_size: Batch size.
        :return sliced file list"""
        batches = math.ceil(len(file_list) / batch_size)
        batch_list: List = []
        for i in range(batches)[0:20]:
            start_batch = i * batch_size
            end_batch = start_batch + batch_size - 1
            batch_list.append(file_list[start_batch:end_batch])
        return batch_list

    def __process_batch(self,
                        file_input: zipfile.ZipFile | gzip.GzipFile | h5.File | tarfile.TarFile | io.BytesIO,
                        file_list: List,
                        process_q: multiprocessing.Queue, batch_process_limit: int = None,
                        batch_chunk_size: int = None) -> None:
        """Process input files in batches calculated from __size_batching to free memory and avoid memory errors.
        Batch limits are set in the hdfs=config.json file.
        :param file_input can be a ZipFile, GzipFile or h5.File
        :param file_list is a list of files in file format returned by self.file_list()"""
        allowed_types = self.__config_dict.get('allowed_types')
        if not batch_chunk_size:
           batch_chunk_size = self.__config_dict.get('batch_chunk_size')
        if not batch_process_limit:
            batch_process_limit = self.__config_dict.get('batch_process_limit')

        if not file_list:
            if self.__input_file.split('.')[-1] in allowed_types:
                self.classify_inputs(file_input, file_input, process_q)
        else:
            batch_list = self.__size_batching(file_list, batch_chunk_size)
            file_sz = len(file_list)
            b_counter = 0
            print(f"Processing {file_sz} files in batches of {batch_process_limit * batch_chunk_size} ")
            for batch in batch_list:
                for file in batch:
                    if not file.endswith('/'):
                      self.classify_inputs(file, file_input, process_q)
                    else:
                       print(f"Skipping directory: {file}")
                       continue
                b_counter += 1
                print(f""" Process data in {batch_process_limit * batch_chunk_size} files per batch, freeing up host memory consumption """)
                print(f"Batch {b_counter} of {len(batch_list)} at {datetime.now()}")

                """ Below when batch counter matches the process_limit, begin writing to h5 file"""
                """ This will clear the contents of the queue and write to the h5 file """
                if b_counter == batch_process_limit:
                    self._data_handler(process_q, file_group=file.split('/')[0], file_list=file_list)
                    b_counter = 0
                    gc.collect()
                elif batch_list[-1] == batch:
                    self._data_handler(process_q, file_group=file.split('/')[0], file_list=file_list)
                    gc.collect()
                    b_counter = 0

    @staticmethod
    def _file_io_buffer(input_file):
        """Create buffer for reading bytes from a file."""
        with open(input_file, 'rb', buffering=(1024 * 1024 * 100)) as file:
            bytes_content = file.read()
            file_buffer = io.BytesIO(bytes_content)
            return file_buffer

    def _input_file_type(self, process_q: multiprocessing.Queue, batch_process_limit: int) -> str | None:
        """Determine the type of input file accordingly. Valid input types are ZIP, HDF5, TAR, and GZIP. Send file to
        self.__process_batch to create a sliced list of files
        :param process_q: multiprocessing.Queue
        :param batch_process_limit: int"""
        try:
            if self.__input_file.endswith('zip' or 'z'):
                file_buffer = self._file_io_buffer(self.__input_file)
                zipped = zipfile.ZipFile(file_buffer, 'r', allowZip64=True)
                file_list = self.file_list(zipped)
                self.__process_batch(zipped, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('h5' or 'hdf5'):
                h5file = h5.File(self.__input_file, 'r')
                self.__process_batch(h5file, [], process_q, batch_process_limit)

            elif self.__input_file.endswith('tar.gz') | self.__input_file.endswith('tar'):
                """Provide custom buffer size for tar files to avoid memory errors."""
                with tarfile.open(self.__input_file, 'r', bufsize=(1024 * 1024 * 500)) as tar:
                    file_list = self.file_list(tar)
                    self.__process_batch(tar, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('gz' or 'gzip'):
                file_buffer = self._file_io_buffer(self.__input_file)
                file_list = self.file_list(file_buffer)
                self.__process_batch(file_buffer, file_list, process_q, batch_process_limit)

            else:
                file_list = []
                file_buffer = self._file_io_buffer(self.__input_file)
                self.__process_batch(file_input=file_buffer, file_list=file_list,
                                     process_q=process_q, batch_process_limit=batch_process_limit)

        except Exception as e:
            print(f'_input_file_type Exception: {e, e.args}')
        except BaseException as be:
            print(f'_input_file_type BaseException: {be, be.args}')

    def _process_schema_input(self, h5_file: h5.File, content_list: List = None) -> Tuple:
        """ Use multiprocessing and queues for large image lists
        :param h5_file is file object
        :param content_list is list
        :returns tuple containing list of files, multiprocessing.Queue, and multiprocessing.Process"""
        try:
            if os.path.exists(self.__schema_file):
                paths = self._schema_processor.process_schema_file(h5_file, content_list)
                process_q, local_mpq = self.start_mp()
                process_q.get()
                return paths, process_q, local_mpq
            else:
                print("No schema file found - using defaults")
        except Exception as e:
            print(f'_process_schema_input Exception: {e}')
        except BaseException as be:
            print(f'_process_schema_input BaseException: {be}')

    def _data_handler(self, process_q: multiprocessing.Queue, file_group: AnyStr, file_list: AnyStr) -> None:
        """While there is content in the queue, process it and write to the HDF5 file.
        If the queue is empty, write the schema to the HDF5 file.
        If the queue is empty and the file list is empty, write the root attributes to the HDF5 file.
        :param process_q: multiprocessing.Queue
        :param file_list: list
        :param file_group: str
        Tuples of lists and files are returned from the queue each iteration."""
        try:
            while not process_q.empty():
               for data in process_q.get():
                  if isinstance(data, List):
                     """data from queue should be tuples"""
                     if len(data) == 1:
                         if len(data[0]) == 3:
                            file_group, content_list, file = data[0]
                            file = file_group
                            file_group = file_group.split('/')[-1]
                         else:
                             file_name = data[0]
                             self.update_file_group(h5_file=h5_file, file_group=file_group,
                                                    attr='file-name', attr_value=file_name)
                     elif len(data) == 3:
                        file_group, content_list, file = data
                        file = file_group
                        file_group = file_group.split('/')[-1]

                     if file_group and content_list and file:
                        h5_file = h5.File(self.__output_file, mode=self.__write_mode)
                        if isinstance(content_list, np.ndarray):
                           self.create_dataset_from_input(h5_file=h5_file, data=content_list,
                                                          file_group=file_group, file=file)
                           self._flush_content_to_file(h5_file=h5_file)

                        elif isinstance(data, Dict) | isinstance(data, List):
                           self.create_dataset_from_input(h5_file=h5_file, data=data, file_group=file_group, file=file)
                           self._flush_content_to_file(h5_file=h5_file)

                     else:
                        if content_list and file_list:
                            for contents, file in content_list, file_list:
                               for file_group, content_lines in contents:
                                  if not self._get_file_group(h5_file=h5_file, group_name=file_group):
                                     self.create_file_group(file_group, len(content_list))
                                     print("write file group attrs")
                                  elif not file_group:
                                     file_group = 'root'
                                     if not self._get_file_group(h5_file=h5_file, group_name=file_group):
                                        self.create_file_group(h5_file=h5_file, group_name=file_group,
                                                               content_size=len(content_list))
                                        print("write root attrs")
                                        self.process_content_list(content=content_lines, file_group=file_group)
                  else:
                     print("Cannot process file fragment --> No content to process")
                     pass

        except Exception as e:
           print(f'_data_handler Exception: {e, e.args}')
           pass
        except BaseException as be:
           print(f'_data_handler BaseException: {be, be.args}')
        finally:
           if process_q.empty():
              exit()

    def start_processing(self, group_keys: List, batch_process_limit: int = None) -> h5.File.keys:
        """Start the data processing pipeline.
        :param group_keys is list of group names
        :param batch_process_limit is the maximum number of splits to create
        :returns h5.File keys"""
        try:
            """h5 file is closed after the init of the class; open it here"""
            h5_file = h5.File(self.__output_file, mode=self.__write_mode)
            paths, process_q, local_mpq = self._process_schema_input(h5_file)
            for gkey in group_keys:
                path = [p for p in paths if p[0].endswith(gkey)]
            else:
                print("No schema file found - using defaults")

            if process_q.empty():
                self._flush_content_to_file(h5_file)
                """ Close the file to avoid locking error when processing input file type.
                The schema write already has the file open."""
                h5_file.close()
                """ File will be opened again below"""
                self._input_file_type(process_q, batch_process_limit)
                h5_file.flush()
                """ flush memory content to disk """
            elif self.__input_dict:
                content_list = [self.__input_dict]
                file_list: List = []
                file_group = 'root'
                process_q.put([file_group, content_list, file_list])
                self._data_handler(file_group=file_group, file_list=file_list, process_q=process_q)

            """ Shutdown of Queue and Process """
            if process_q.empty():
                local_mpq.current_process.close()

        except Exception as e:
            print(f'start_processor Exception: {e, e.args}')
        except BaseException as be:
            print(f'start_processor BaseException: {be, be.args}')
            pass
        finally:
            """ Open the file again to read the keys"""
            h5_file = h5.File(self.__output_file, 'r')
            h5_keys = h5_file.keys()
            """ Close H5 file again when finished; return keys """
            h5_file.close()
            return h5_keys
