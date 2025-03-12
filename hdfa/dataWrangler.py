import gc
import io
import math
import time
import signal
from parsingProcessor import ParsingProcessor
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
from typing import Any, AnyStr, Dict, List, Tuple
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
        Initialize the H5DataProcessor class.
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
        self.__h5_file = self._create_file(self.__output_file)
        self.__schema_dict = json.load(open(self.__schema_file, 'r'))
        self.__config_dict = json.load(open(self.__config_file, 'r'))
        """ Initialize image, video, schema, parsing, and text processors """
        self._img_processor = ImageProcessor(self.__output_file,
                                             schema_dict=self.__schema_dict,
                                             config_dict=self.__config_dict)
        self._vid_processor = VideoProcessor()
        self._txt_processor = TextFileProcessor()
        self._schema_processor = SchemaProcessor(output_file=self.__output_file,
                                                 schema_file=self.__schema_file,
                                                 config_file=self.__config_file
        )
        self._parsing_processor = ParsingProcessor()

    @staticmethod
    def random_int_generator() -> str:
        random_int = random.randint(1, 1000000)
        return str(random_int)

    @classmethod
    def _create_file(cls, __output_file: AnyStr) -> h5.File:
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
        gc.collect()

    @staticmethod
    def gen_dataset_name():
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

            self._flush_content_to_file(h5_file=h5_file)
        except AttributeError as ae:
            print(f'AttributeError: {ae, ae.args}')
        except Exception as e:
            print(f'Attribute Exception: {e, e.args}')

    def _process_list_depth(self, data: List) -> List | None:
        list_depth = self._parsing_processor.find_list_depth(data)
        result: List = []
        x = 0
        while x < list_depth:
           for row in data:
              result.append(self._parsing_processor.process_row(row))
              x += 1
        return result

    def create_dataset_from_dict(self, h5_file: h5.File, data: Dict, file_group: AnyStr) -> None:
        """Create a dataset from a dictionary. Dictionaries are string-ified into key:value lists. Nested dictionaries become nested lists
        :param h5_file: HDF5 file object.
        :param data: HDF5 dataset data in dictionary format.
        :param file_group: HDF5 group.
        :return: None"""
        try:
            kv_list = self._parsing_processor.parse_data(input_dict=data)
            obj, req, d = kv_list
            g = h5_file.require_group(file_group)
            name = self.gen_dataset_name()
            g.require_group(name)
            data_str = self._process_list_depth(data=d)
            for row in data_str:
               if row and row != []:
                  name = self.gen_dataset_name()
                  g.create_dataset(name=name, data=row, shape=len(row), compression='gzip')
                  print(f"write dataset {name} in file group {file_group}")
            self._flush_content_to_file(h5_file=h5_file)
        except ValueError as ve:
            print(f'Dataset Dict ValueError: {ve, ve.args}')
        except TypeError as te:
            print(f'Dataset Dict TypeError: {te, te.args}')
        except Exception as e:
            print(f'Dataset Dict Exception: {e, e.args}')

    def create_dataset_from_input(self, h5_file: h5.File, data: List | Dict | np.ndarray, file_group: AnyStr,
                                  file: AnyStr) -> None:
        """Create a dataset from a list, dictionary, or Numpy array. Applying updates to group attributes.
        :param h5_file: HDF5 file object.
        :param data: HDF5 dataset data in list, dictionary, or numpy array.
        :param file_group: HDF5 group.
        :param file: file name from input source.
        :return: None"""
        try:
            self._flush_content_to_file(h5_file=h5_file)
            if isinstance(data, List) and isinstance(data[0], List | np.ndarray | Dict):
               if isinstance(data[0], Dict):
                  self.create_dataset_from_dict(h5_file=h5_file, data=data[0], file_group=file_group)
                  self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

               elif isinstance(data[0], np.ndarray):
                   name = self.gen_dataset_name()
                   g = h5_file.require_group(file_group)
                   g.create_dataset(name=name, data=data[0], compression='gzip')
                   print(f"write dataset {name} - {file}")
                   self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

               elif isinstance(data[0], List):
                  data_str = self._process_list_depth(data=data)
                  name = self.gen_dataset_name()
                  g = h5_file.require_group(file_group)
                  if data_str and not data_str == [[]]:
                     g.create_dataset(name=name, data=data_str, compression='gzip')
                     self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)
                     print(f"write dataset {name} - {file}")
                  else:
                     print(f"No data to write for {file}")

            elif isinstance(data, Dict):
               self.create_dataset_from_dict(h5_file=h5_file, data=data, file_group=file_group)
               self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

            elif isinstance(data, np.ndarray):
               name = self.gen_dataset_name()
               g = h5_file.require_group(file_group)
               g.create_dataset(name=name, data=data, compression='gzip')
               print(f"write dataset {name} - {file}")
               self.set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

            return self._flush_content_to_file(h5_file=h5_file)
        except ValueError as ve:
            print(f'Dataset Input ValueError: {ve, ve.args}')
        except TypeError as te:
            print(f'Dataset input TypeError: {te, te.args}')
        except Exception as e:
            print(f'Dataset Input Exception: {e, e.args}')

    def create_file_group(self, h5_file: h5.File, group_name: AnyStr, content_size: int = 0) -> None:
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
        video(MP4/MP3), and image files including PNG, JPEG, BMP, TIFF, SVG, BMP, GIF, and ICO.
        Add results to multiprocessing.Queue.
        :param file: Path to the input file in archive (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, Tarfile, h5.File, and BytesIO object.
        :param process_q: multiprocessing.Queue
        :return: None"""
        content_list: List = []
        processed_file_list: List[str] = []
        image_extensions = self.__config_dict.get('image_extensions')
        video_extensions = self.__config_dict.get('video_extensions')
        if isinstance(open_file, io.BytesIO):
            io_file = open_file.getbuffer()
            if max(io_file.suboffsets) >= 1:
                """It has PIL file properties, so convert to JPEG"""
                file = open_file.name + '.jpeg'
                open_file.close()
                open_file = self._file_io_buffer(file)
                file = open_file.name
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
            process_q.put(f"classify_inputs Exception: {e, e.args}")
            pass
        finally:
            gc.collect()

    def process_content_list(self, content: List, file_group: AnyStr, h5_file: h5.File = None):
        try:
            for file, line in content:
                self.create_dataset_from_input(h5_file=h5_file, data=line, file_group=file_group, file=file)
                self._flush_content_to_file(h5_file=h5_file)
        except Exception as e:
            print(f'process_content_list Exception: {e}')

    def start_mp(self) -> [multiprocessing.Queue, multiprocessing.Process]:
        """ Use multiprocessing and queues for large number of files and content; requires batch limits to be set in order to control
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
            process_q.put(f"start_mp Exception: {e, e.args}")

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
    def _size_batching(file_list: List, batch_size: int):
        """Compute and slice Batch file list into a specified size for processing.
        :param file_list: List of files to process.
        :param batch_size: Batch size.
        :return sliced file list"""
        batches = math.ceil(len(file_list) / batch_size)
        batch_list: List = []
        for i in range(batches):
            start_batch = i * batch_size
            end_batch = start_batch + batch_size - 1
            batch_list.append(file_list[start_batch:end_batch])
        return batch_list

    def _process_batch(self,
                       file_input: zipfile.ZipFile | gzip.GzipFile | h5.File | tarfile.TarFile | io.BytesIO,
                       file_list: List,
                       process_q: multiprocessing.Queue, batch_process_limit: int = None,
                       batch_chunk_size: int = None) -> None:
        """Process input files in batches calculated from __size_batching to manage memory consumption.
        Batch limits are set in the hdfs=config.json file.
        :param file_input can be a ZipFile, GzipFile, Tarfile, h5.File, or BytesIO object.
        :param file_list is a list of files in file format returned by self.file_list()"""
        try:
            allowed_types = self.__config_dict.get('allowed_types')
            if not batch_chunk_size:
               batch_chunk_size = self.__config_dict.get('batch_chunk_size')
            if not batch_process_limit:
                batch_process_limit = self.__config_dict.get('batch_process_limit')

            if not file_list:
                if self.__input_file.split('.')[-1] in allowed_types:
                    self.classify_inputs(file_input, file_input, process_q)
            else:
                batch_list = self._size_batching(file_list, batch_chunk_size)
                file_sz = len(file_list)
                b_counter = 0
                f_counter = 0
                print(f"Processing {file_sz} files in batches of {batch_process_limit * batch_chunk_size} ")
                for batch in batch_list:
                    for file in batch:
                        if not file.endswith('/') and file.split('.')[-1] in allowed_types:
                          f_counter += 1
                          print(f""" Processing file {file}, file count {f_counter} in slice {len(batch)} in batch {b_counter} """)
                          self.classify_inputs(file, file_input, process_q)
                        else:
                           print(f"Skipping: {file}")
                           continue
                    b_counter += 1
                    print(f""" Process data in {batch_process_limit * batch_chunk_size} files per batch, managing host memory consumption """)
                    print(f"Batch {b_counter} of {len(batch)} at {datetime.now()}")

                    """ Below when batch counter matches the process_limit, begin writing to h5 file"""
                    """ This will clear the contents of the queue and write to the h5 file """
                    if b_counter == batch_process_limit:
                        self.data_handler(process_q, file_group=file.split('/')[0], file_list=file_list)
                        b_counter = 0
                    elif batch_list[-1] == batch:
                        self.data_handler(process_q, file_group=file.split('/')[0], file_list=file_list)
                        b_counter = 0
                total_batch_count = b_counter + 1
                print(f"Total batches processed: {total_batch_count}")
                resource_check = total_batch_count % 10
                if (str(resource_check).split('.')[-1] == 0) and (total_batch_count > 0):
                    print(f"Total batches processed: {total_batch_count}")
                    print(f"Total files processed: {file_sz}")
                    print(f"Total files in queue: {process_q.qsize()}")
                    print(f"Total files in h5 file: {len(self._get_file_group(file_input, file.split('/')[0]))}")
                    print(f"Total bytes in input file: {os.path.getsize(self.__input_file)}")
                    print(f"Total bytes in queue: {process_q.qsize() * 1024}")
                    print(f"Total bytes in h5 file: {self._get_file_group(file_input, file.split('/')[0]).attrs['content_size'] * 1024}")
                    print(f"Total bytes in input file: {os.path.getsize(self.__input_file) * 1024}")

        except Exception as e:
           print(f'__process_batch Exception: {e, e.args}')
           process_q.put(f" __process_batch Exception: {e, e.args}")
        finally:
           gc.collect()

    @staticmethod
    def _file_io_buffer(input_file: AnyStr) -> io.BytesIO:
        """Create buffer for reading bytes from a file."""
        with open(input_file, 'rb', buffering=(1024 * 1024 * 400)) as file:
            bytes_content = file.read()
            file_buffer = io.BytesIO(bytes_content)
            return file_buffer

    def input_file_type(self, process_q: multiprocessing.Queue, batch_process_limit: int) -> str | None:
        """Determine the type of input file accordingly. Valid input types are ZIP, HDF5, TAR, and GZIP. Send file to
        self.__process_batch to create a sliced list of files
        :param process_q: multiprocessing.Queue
        :param batch_process_limit: int"""
        try:
            if self.__input_file.endswith('zip' or 'z'):
                file_buffer = self._file_io_buffer(self.__input_file)
                zipped = zipfile.ZipFile(file_buffer, 'r', allowZip64=True)
                file_list = zipped.namelist()
                self._process_batch(zipped, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('h5' or 'hdf5'):
                h5file = h5.File(self.__input_file, 'r')
                self._process_batch(h5file, [], process_q, batch_process_limit)

            elif self.__input_file.endswith('tar.gz') | self.__input_file.endswith('tar'):
                """Provide custom buffer size for tar files to get more efficient reads."""
                with tarfile.open(self.__input_file, 'r', bufsize=(1024 * 1024 * 400)) as tar:
                    file_list = self.file_list(tar)
                    self._process_batch(tar, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('gz' or 'gzip'):
                file_buffer = self._file_io_buffer(self.__input_file)
                file_list = self.file_list(file_buffer)
                self._process_batch(file_buffer, file_list, process_q, batch_process_limit)

            else:
                """If input file is not a ZIP, HDF5, TAR, or GZIP file, process the entire file as a single file."""
                file_list = []
                file_buffer = self._file_io_buffer(self.__input_file)
                self._process_batch(file_input=file_buffer, file_list=file_list,
                                    process_q=process_q, batch_process_limit=batch_process_limit)

        except Exception as e:
            print(f'_input_file_type Exception: {e, e.args}')
            process_q.put(f"_input_file_type Exception: {e, e.args}")
        except BaseException as be:
            print(f'_input_file_type BaseException: {be, be.args}')
            process_q.put(f"_input_file_type BaseException: {be, be.args}")

    def _process_schema_input(self, h5_file: h5.File, content_list: List = None) -> Tuple[
	                                                                                    None | list | str, Any, Any] | None:
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
            process_q.put(f"_process_schema_input Exception: {e}")
        except BaseException as be:
            print(f'_process_schema_input BaseException: {be}')
            process_q.put(f"_process_schema_input BaseException: {be}")

    def process_ds_default(self, h5_file: h5.File, file_group: AnyStr, file: AnyStr, content_list: List = None) -> None:
        file_group = file_group.split('/')[-1]
        self.create_dataset_from_input(h5_file=h5_file, data=content_list, file_group=file_group, file=file)
        self._flush_content_to_file(h5_file=h5_file)

    def data_handler(self, process_q: multiprocessing.Queue, file_group: AnyStr, file_list: AnyStr) -> None:
        """While there is content in the queue, process it and write to the HDF5 file.
        If the queue is empty, write the schema to the HDF5 file.
        If the queue is empty and the file list is empty, write the root attributes to the HDF5 file.
        :param process_q: multiprocessing.Queue
        :param file_list: list
        :param file_group: str
        Tuples of lists and files are returned from the queue each iteration."""
        try:
            h5_file = h5.File(self.__output_file, mode=self.__write_mode)
            while not process_q.empty():
               for data in process_q.get():
                  if isinstance(data, List):
                     """data from queue should be tuples"""
                     if isinstance(data[0][1], np.ndarray):
                        file_group, content_list, file = data[0]
                        self.process_ds_default(h5_file=h5_file, file_group=file_group,
                                                file=file, content_list=content_list)
                     elif (isinstance(data, List) and not isinstance(data[0][1], np.ndarray) and len(data[0]) == 3):
                        file_group, content_list, file = data[0]
                        file = file_group
                        self.process_ds_default(h5_file=h5_file, file_group=file_group,
                                                file=file, content_list=content_list)
                     else:
                        file_name = data[0]
                        self.update_file_group(h5_file=h5_file, file_group=file_group,
                                               attr='file-name', attr_value=file_name)

                        if not isinstance(content_list, np.ndarray) and content_list and file_list and not file_group:
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
           process_q.put(f"_data_handler Exception: {e, e.args}")
           pass
        except BaseException as be:
           print(f'_data_handler BaseException: {be, be.args}')
           process_q.put(f"_data_handler BaseException: {be, be.args}")
        finally:
           if process_q.empty():
              pass

    def start_processors(self, group_keys: List, batch_process_limit: int = None) -> h5.File.keys:
        """Start the data processing pipeline.
        :param group_keys is list of group names
        :param batch_process_limit is the maximum number of splits to create
        :returns h5.File keys"""
        try:
            signal.signal(signal.SIGTERM, self.signal_handler)
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
                self.input_file_type(process_q, batch_process_limit)
                h5_file.flush()
                """ flush memory content to disk """
            elif self.__input_dict:
                content_list = [self.__input_dict]
                file_list: List = []
                file_group = 'root'
                process_q.put([file_group, content_list, file_list])
                self.data_handler(file_group=file_group, file_list=file_list, process_q=process_q)

            """ Shutdown of Queue and Process """
            if process_q.empty():
                local_mpq.current_process.close()

        except Exception as e:
            print(f'start_processor Exception: {e, e.args}')
            process_q.put(f"start_processor Exception: {e, e.args}")
            pass
        except BaseException as be:
            print(f'start_processor BaseException: {be, be.args}')
            process_q.put(f"start_processor BaseException: {be, be.args}")
            pass
        finally:
            """ Open the file again to read the keys"""
            h5_file = h5.File(self.__output_file, 'r')
            h5_keys = h5_file.keys()
            """ Close H5 file again when finished; return keys """
            h5_file.close()
            return h5_keys

    @staticmethod
    def signal_handler(sig, frame, h5_file: h5.File) -> None:
        """Handle signals sent to the process on Mac or Linux.
        :param sig: signal
        :param frame: frame
        :return: None"""
        print('Received signal:', sig)
        print('Performing cleanup...')
        # Add cleanup code here, e.g., closing files, releasing resources
        gc.collect()
        h5_file.flush()
        time.sleep(1)  # Simulate cleanup
        print('Cleanup complete. Exiting.')
        pass

