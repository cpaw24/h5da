import gc
import io
import math
import multiprocessing
import os
import uuid
import tarfile
import zipfile
from hdfa.mpLocal import MpQLocal
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
import h5rdmtoolbox as h5tbx


class DataProcessor:
    def __init__(self, output_file: AnyStr, input_file: AnyStr,
                 input_dict: Dict | np.ndarray = None, write_mode: AnyStr = 'a',
                 schema_file: AnyStr = None, config_file: AnyStr = None) -> None:
        """
        Initialize the H5DataCreator class.
        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz, .h5).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        :param write_mode: Optional mode to write the HDF5 file. Default is 'a'.
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

    @staticmethod
    def random_int_generator() -> str:
        random_int = random.randint(1, 1000000)
        return str(random_int)

    @classmethod
    def __create_file(cls, __output_file: AnyStr) -> h5.File:
        """Create an HDF5 file. Using customized page size and buffer size to reduce memory usage."""
        """ Set the kwargs values for the HDF5 file creation."""
        kwarg_vals={'libver': 'latest', 'driver': 'core', 'backing_store': True, 'fs_persist': True,
                  'fs_strategy': 'page', 'fs_page_size': 65536, 'page_buf_size': 655360}

        with h5tbx.File(__output_file, mode='w', **kwarg_vals) as __h5_file:
           return __h5_file

    @staticmethod
    def __write_content_to_file(h5_file: h5.File) -> None:
        """Flush content from memory to HDF5 file."""
        h5_file.flush()

    def __gen_dataset_name(self):
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

    def __set_dataset_attributes(self, h5_file: h5.File, file_group: AnyStr, file: AnyStr) -> None:
        """Standard set of attributes to write per dataset.
        :param h5_file: HDF5 file object.
        :param file_group: HDF5 group.
        :param file: input file from archive.
        :return: None"""
        try:
            self.update_file_group(h5_file=h5_file, file_group=file_group,
                                   attr='source-file-name', attr_value=file.split('/')[-1])
            self.update_file_group(h5_file=h5_file, file_group=file_group,
                                   attr='file-type-extension', attr_value=file.split('.')[-1])
            self.update_file_group(h5_file=h5_file, file_group=file_group,
                                   attr='source-root', attr_value=file.split('/')[0])
        except AttributeError as ae:
            print(f'AttributeError: {ae}')
        except Exception as e:
            print(f'Attribute Exception: {e}')

    def create_dataset_from_dict(self, h5_file: h5.File, name: AnyStr, data: Dict, file_group: AnyStr) -> None:
        """Create a dataset from a dictionary. Numpy arrays are string-ified and converted to byte strings.
        :param h5_file: HDF5 file object.
        :param name: HDF5 dataset name.
        :param data: HDF5 dataset data in dictionary format.
        :param file_group: HDF5 group.
        :return: None"""
        try:
            kv_list = self.parse_data(input_dict=data)
            kva = np.array(kv_list, dtype=np.dtype('S'))
            h5_file.require_group(file_group).create_dataset(name, data=kva, compression='gzip')

        except ValueError as ve:
            print(f'Dataset from Dict ValueError: {ve.args}')
        except TypeError as te:
            print(f'Dataset from Dict TypeError: {te.args}')
        except Exception as e:
            print(f'Dataset from Dict Exception: {e}')

    def create_dataset_from_input(self, h5_file: h5.File, data: List | Dict | np.ndarray, file_group: AnyStr, file: AnyStr) -> None:
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
                init_byte = data[0]
                if isinstance(init_byte, int):
                    kva = np.array(data, dtype=np.dtype('I'))
                elif isinstance(init_byte, str):
                    kva = np.array(data, dtype=np.dtype('S'))
                    name = self.__gen_dataset_name()
                    h5_file.require_group(file_group).create_dataset(name=name, data=kva, compression='gzip')
                    self.__set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)
                    print(f"write dataset {name} - {file}")

            elif isinstance(data, List):
                list_count = 0
                name = self.__gen_dataset_name()
                if isinstance(data[0], Dict):
                    for l in data:
                        idx = list_count
                        name = name + '-' + str(idx)
                        self.create_dataset_from_dict(name=name, data=l, file_group=file_group)
                        self.__set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)
                        print(f"write dataset {name} - {file}")
                        list_count += 1

            elif isinstance(data, Dict):
                name = self.__gen_dataset_name()
                self.create_dataset_from_dict(name=name, data=data, file_group=file_group)
                print(f"write dataset {name} - {file}")
                self.__set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

            elif isinstance(data, np.ndarray):
                h5_file.flush()
                name = self.__gen_dataset_name()
                h5_file.require_group(file_group).create_dataset(name=name, data=data, compression='gzip')
                print(f"write dataset {name} - {file}")
                self.__set_dataset_attributes(h5_file=h5_file, file_group=file_group, file=file)

            gc.collect()
            return h5_file.flush()
        except ValueError as ve:
            print(f'Dataset from Input ValueError: {ve.args}')
        except TypeError as te:
            print(f'Dataset from input TypeError: {te.args}')
        except Exception as e:
            print(f'Dataset from Input Exception: {e}')

    @staticmethod
    def process_json(file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                       content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]] | None:
        """Converts JSON files to dictionaries
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return tuple[content_list, processed_file_list]"""
        try:
            file_name = file.casefold()
            raw_content = open_file.read(file).decode('utf-8').splitlines()
            content = [row for row in raw_content]
            content = json.loads(content[0])
            line_count = len(content_list)
            content_list.append([file_name, content, line_count])
            processed_file_list.append(file_name + '-' + str(line_count))
            return content_list, processed_file_list
        except Exception as e:
            print(f'Exception: {e}')

    @staticmethod
    def process_csv(file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]] | None:
        """Converts CSV files to lists of values
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return tuple[content_list, processed_file_list]"""
        try:
            with open_file.open(file, force_zip64=True) as csv_file:
                csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
                                        doublequote=True, quotechar='"')
                content = [row for row in csv_reader]
                content_size = len(content)
                content_list.append([file, content, content_size])
                processed_file_list.append(file)
                return content_list, processed_file_list

        except FileNotFoundError as fnf_error:
            print(f'FileNotFound: {fnf_error.args}')
        except csv.Error as csv_error:
            print(f'CSV Error" {csv_error.args}')
        except Exception as e:
            print(f'process_csv Exception: {e}')

    def process_svg(self, file: AnyStr, open_file: zipfile.ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile,
                      content_list: List, processed_file_list: List[str]) -> [List, List]:
        """Converts SVG files to Numpy arrays
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return tuple[content_list, processed_file_list]"""
        ds = file.split('/')[0]
        target_file = self.__schema_dict.get(self.__output_file.split('/')[0])
        l_members = len(target_file)
        if l_members != 0:
          for i in range(l_members):
               target_file_name = target_file[i + 1]
        temp_img = f"{self.random_int_generator()}_temp.png"
        try:
            drawing = svg2rlg(open_file.open(file, force_zip64=True))
            renderPM.drawToFile(drawing, temp_img, fmt='PNG')
            img = np.array(Image.open(temp_img))
            content_list.append([ds, img, file])
            processed_file_list.append(file)
            return content_list, processed_file_list

        except Exception as e:
            print(f' process_svg Exception: {e}')
            if os.path.exists(temp_img):
                os.remove(temp_img)
        finally:
            """ Ensure temp file is removed. """
            if os.path.exists(temp_img):
                os.remove(temp_img)

    @staticmethod
    def process_video(file: AnyStr, open_file: zipfile.ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile,
                        content_list: List, processed_file_list: List[str]) -> List:
        """Converts MP4/MP3 video to Numpy array
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return list[content_list, processed_file_list]"""
        try:
            import cv2
            ds = file.split('/')[0]
            frames = []
            with open_file.read(file) as video:
               with cv2.VideoCapture(video) as cap:
                  while cap.isOpened():
                     ret, frame = cap.read()
                     if not ret:
                        break
                     frames.append(frame)

                  cap.release()
                  final_frame = np.dstack(frames)
                  content_list.append([ds, final_frame, file])
                  processed_file_list.append(file)
                  return [content_list, processed_file_list]
        except Exception as e:
            print(f'process_video Exception: {e}')

    def process_bio_image(self):
        pass

    def create_file_group(self, h5_file: h5.File, group_name: AnyStr, content_size: int = 0):
        """Create a new file group in the HDF5 file.
        :param h5_file: HDF5 file object.
        :param group_name: New file group name.
        :param content_size: New file group content size.
        :return None"""
        created = self.__get_file_group(h5_file, group_name)
        if not created:
            h5_file.create_group(group_name, track_order=True)
            h5_file[group_name].attrs['file_name'] = h5_file.name
            h5_file[group_name].attrs['schema_file'] = self.__schema_file
            h5_file[group_name].attrs['content_size'] = content_size
            print(f"write group {group_name} attrs")
            return h5_file.clear()

    @staticmethod
    def __get_file_group(h5_file: h5.File, group_name: AnyStr):
        return h5_file.get(group_name)

    def __write_schema(self, h5_file: h5.File, path: List,
                       content_list: List, processed_file_list: List[str]):
        try:
            flg: AnyStr = ''
            if not content_list: content_list: List = []
            content_length = content_list.count('unknown') | len(content_list)
            if not processed_file_list: processed_file_list: List = []
            for l in path:
                if isinstance(l, List):
                    for i in l:
                        flg = i
                else:
                    flg = l

                self.create_file_group(h5_file, flg, content_length)
                content_list.append(['Group', flg])
                processed_file_list.append(flg)
            return content_list, processed_file_list

        except Exception as e:
            print(f'__write_schema Exception: {e}')

    def convert_images(self, process_q: multiprocessing.Queue, file: AnyStr,
                         open_file: zipfile.ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile, content_list: List,
                         processed_file_list: List[str]):
        """Convert image files to Numpy arrays and add them to multiprocessing.Queue.
        :param process_q: multiprocessing.Queue
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return None"""
        try:
            ds = file.split('/')[0]
            """ Process SVG files (Convert to numpy array via PNG generation) """
            if file.endswith('svg'):
               try:
                  content_list, processed_file_list = self.process_svg(file, open_file,
                                                                       content_list, processed_file_list)
                  process_q.put(content_list)
               except Exception as e:
                  print(e)

            if file.endswith('tiff'):
               with open_file.open(file) as tiff_file:
                  img = tifffile.imread(tiff_file)
                  content_list.append([ds, img, tiff_file])
                  process_q.put(content_list)
            else:
                with open_file.open(file) as img_file:
                    img = np.array(Image.open(img_file))
                    content_list.append([ds, img, file])
                    process_q.put(content_list)

        except Exception as e:
           print(f'convert_images Exception: {e}')

    def __classify_inputs(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile,
                          process_q: multiprocessing.Queue) -> None:
        """Classify and process input file content into structured data formats. Valid inputs types are JSON, CSV,
        video(MP4/MP3), and image files including PNG, JPEG, BMP, TIFF, SVG, BMP, GIF, and ICO. Add them to multiprocessing.Queue.
        :param file: Path to the input file in archive (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param process_q: multiprocessing.Queue"""
        content_list: List = []
        processed_file_list: List[str] = []
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'blp', 'bufr', 'cur', 'dcx', 'dib', 'eps', 'fits',
                            'gif', 'jfif', 'jpe', 'ico', 'icns', 'j2c', 'j2k', 'jp2', 'jpc', 'jpx', 'pcx', 'pxr',
                            'pgm', 'ppm', 'sgi', 'svg')
        video_extensions = ('mpeg', 'mp4', 'mp3')
        try:
            if not file.endswith('/'):
                """ Process JSON files """
                if file.endswith('json') or file.endswith('jsonl'):
                    content_list, processed_file_list = self.process_json(file, open_file,
                                                                            content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])
                    """ Process image files """
                elif file.endswith(image_extensions):
                    self.convert_images(process_q, file, open_file,
                                          content_list, processed_file_list)
                    """ Process CSV files """
                elif file.endswith('csv'):
                    content_list, processed_file_list = self.process_csv(file, open_file,
                                                                           content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])
                    """ Process video files """
                elif file.endswith(video_extensions):
                    content_list, processed_file_list = self.process_video(file, open_file,
                                                                             content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])

        except Exception as e:
            print(f'__classify_inputs Exception: {e.__cause__}')

    def parse_data(self, input_dict: Dict | np.ndarray) -> List:
        """
        Recursively parses a nested dictionary or a numpy array to extract and organize
        data into a list of key-value pairs.
        :param input_dict: Dictionary or numpy array to parse.
        :return: List of key-value pairs, numpy arrays, or possibly tuples.
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
        elif isinstance(input_dict, np.ndarray):
            """ If the input is a numpy array, append it directly """
            value_list.append([input_dict])
        elif isinstance(input_dict, Tuple):
            _a, _b, _c = input_dict
            value_list.append([_c, (_a, _b)])
        return value_list

    def __process_content_list(self, content: List, file_group: AnyStr, h5_file: h5.File = None):
        try:
            for file, line in content:
                self.create_dataset_from_input(h5_file=h5_file, data=line, file_group=file_group, file=h5_file.filename)
                self.__write_content_to_file(h5_file=h5_file)
        except Exception as e:
            print(f'__process_content_list Exception: {e.__cause__}')
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
           print(f'start_mp Exception: {e, e.__context__}')

    def file_list(self) -> List:
        """Get a list of files from ZIP or GZIP files."""
        if self.__input_file.endswith('zip' or 'z'):
            valid = zipfile.is_zipfile(self.__input_file)
            if valid:
                with ZipFile(self.__input_file, 'r') as zip:
                    file_list = zip.namelist()
                    return file_list
        elif self.__input_file.endswith('tar' or 'tar.gz'):
            with tarfile.open(self.__input_file, 'r') as tar:
                file_list = tar.getnames()
                return file_list
        elif self.__input_file.endswith('gz' or 'gzip'):
            with gzip.open(self.__input_file, 'rb') as gzf:
                file_list = gzip.GzipFile(fileobj=gzf.fileobj.__dict__.get('namelist'))
                return file_list

    @staticmethod
    def __size_batching(file_list: List, batch_size: int):
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

    def __process_batch(self, file_input: zipfile.ZipFile | gzip.GzipFile | h5.File | tarfile.TarFile, file_list: List,
                        process_q: multiprocessing.Queue, batch_process_limit: int, batch_chunk_size: int = 1000) -> None:
        """Process input files in batches calculated from __size_batching to free memory and avoid memory errors.
        Batch limits are set in the hdfs=config.json file.
        :param file_input can be a ZipFile, GzipFile or h5.File
        :param file_list is a list of files in file format returned by self.file_list()"""
        batch_list = self.__size_batching(file_list, batch_chunk_size)
        b_counter = 0
        print(f"Processing {batch_chunk_size} files in batches of 1000")
        for batch in batch_list:
            for file in batch:
                if not file.endswith('/'):
                    if type(file) == tarfile.TarFile:
                       file = tarfile.TarFile.open(file)
                    self.__classify_inputs(file, file_input, process_q)
                else:
                    print(f"Skipping directory: {file}")
                    continue
            b_counter += 1
            f""" Process data in {batch_process_limit} batches freeing up host memory consumption for Queue """
            if b_counter == batch_process_limit:
                self.__data_handler(process_q, file_group=file.split('/')[0], file_list=file_list)
                b_counter = 0
                gc.collect()

    @staticmethod
    def __file_io_buffer(input_file):
        with open(input_file, 'rb', buffering=(1024*1024*1024)) as file:
            bytes_content = file.read()
            file_buffer = io.BytesIO(bytes_content)
            return file_buffer

    def __input_file_type(self, process_q: multiprocessing.Queue, batch_process_limit: int) -> str | None:
        """Determine the type of input file accordingly. Valid input types are ZIP, HDF5, and GZIP. Send file to
        self.__process_batch to create a sliced list of files
        :param process_q: multiprocessing.Queue
        :param batch_process_limit: int"""
        try:
            file_buffer = self.__file_io_buffer(self.__input_file)
            if self.__input_file.endswith('zip' or 'z'):
                zipped = zipfile.ZipFile(file_buffer, 'r', allowZip64=True)
                file_list = self.file_list()
                self.__process_batch(zipped, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('h5' or 'hdf5'):
               h5file = self.__input_file.h5.open().read().decode('utf-8')
               self.__process_batch(h5file, h5file, process_q, batch_process_limit)

            elif self.__input_file.endswith('tar'):
               with tarfile.open(self.__input_file, 'r', bufsize=(1024*1024*24)) as tar:
                  file_list = self.file_list()
                  self.__process_batch(tar, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('tar.gz'):
               with tarfile.open(self.__input_file, 'r:gz', bufsize=(1024*1024*24)) as tar:
                  file_list = self.file_list()
                  self.__process_batch(tar, file_list, process_q, batch_process_limit)

            elif self.__input_file.endswith('gz' or 'gzip'):
                gzfile = gzip.open(file_buffer, 'r')
                file_list = self.file_list()
                self.__process_batch(gzfile, file_list, process_q, batch_process_limit)
            else:
                return 'unknown'

        except Exception as e:
            print(f'__input_file_type Exception: {e}')
        except BaseException as be:
            print(f'__input_file_type BaseException: {be}')


    def __process_schema_file(self, h5_file: h5.File, content_list: List = None,
                            processed_file_list: List = None) -> None | List | str:
        """Load and persist contents of the schema file
        :param h5_file is file object
        :param content_list is list of objects
        :param processed_file_list is list of objects"""
        try:
            if not content_list: content_list: List = []
            if not processed_file_list: processed_file_list: List = []
            """ get schemas """
            h5_file.require_group('root')
            in_path, out_path = self.__get_schemas()
            """ check for output schema """
            if len(out_path) > 0:
                self.__write_schema(h5_file=h5_file, path=out_path,
                                    content_list=content_list, processed_file_list=processed_file_list)
                return out_path
            elif len(in_path) > 0 and len(out_path) == 0:
                self.__write_schema(h5_file=h5_file, path=in_path,
                                    content_list=content_list, processed_file_list=processed_file_list)
                return in_path
            else:
                return "No schema to process"
        except Exception as e:
            print(f'__process_schema Exception: {e.__cause__}')

    def __get_schemas(self) -> Tuple[List, List]:
        """Get input and output schemas from schema file.
        :returns tuple of input and output paths from schema file"""
        in_path_list = []
        out_path_list = []
        schema_file = self.__schema_file
        files = ['outgoing']
        schema = json.load(open(schema_file, 'r'))
        schema_file = schema.get('files')
        for file in files:
            file_path = schema_file.get(file)
            if file == 'outgoing':
               file_name = file_path.get(self.__output_file.split('/')[-1])
               output_groups = file_name.get('groups')
               if output_groups:
                  for k, v in output_groups.items():
                     out_path_list.append([v])

        return in_path_list, out_path_list

    def __process_schema_input(self, h5_file: h5.File, content_list: List = None) -> Tuple:
        """ Use multiprocessing and queues for large image lists
        :param h5_file is file object
        :param content_list is list
        :returns tuple containing list of files, multiprocessing.Queue, and multiprocessing.Process"""
        if os.path.exists(self.__schema_file):
            paths = self.__process_schema_file(h5_file, content_list)
            process_q, local_mpq = self.start_mp()
            process_q.get()
            return paths, process_q, local_mpq
        else:
            print("No schema file found - using defaults")

    def __data_handler(self, process_q: multiprocessing.Queue, file_group: AnyStr, file_list: AnyStr) -> None:
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
                        """data from queue should be a list of many tuples"""
                        file_group, content_list, file = data
                        schema_obj = self.__schema_dict
                        image_types = schema_obj['leaf_content']['images']['sources']
                        vid_types = schema_obj['leaf_content']['videos']['sources']
                        h5_file = h5.File(self.__output_file, mode=self.__write_mode)
                        if isinstance(content_list, np.ndarray):
                            if file.endswith(image_types):
                                file_group = schema_obj['files']['outgoing'][self.__output_file.split('/')[-1]]['FG1']
                            elif file.endswith(vid_types):
                                file_group = schema_obj['files']['outgoing'][self.__output_file.split('/')[-1]]['FG3']
                            self.create_dataset_from_input(h5_file=h5_file, data=content_list,
                                                             file_group=file_group, file=file)
                            self.__write_content_to_file(h5_file=h5_file)

                        elif isinstance(data, Dict):
                            self.create_dataset_from_input(h5_file=h5_file, data=data, file_group=file_group, file=file)
                            self.__write_content_to_file(h5_file=h5_file)

                        else:
                            for contents, file in content_list, file_list:
                                for file_group, content_lines in contents:
                                    if not self.__get_file_group(h5_file=h5_file, group_name=file_group):
                                        self.create_file_group(file_group, len(content_list))
                                        print("write file group attrs")
                                    elif not file_list:
                                        file_group = 'root'
                                        if not self.__get_file_group(h5_file=h5_file, group_name=file_group):
                                            self.create_file_group(h5_file=h5_file, group_name=file_group,
                                                                     content_size=len(content_list))
                                            print("write root attrs")
                                        self.__process_content_list(content=content_lines, file_group=file_group)
                    else:
                        self.__process_content_list(content=content_list, file_group=file_group)

        except Exception as e:
            print(f'__data_handler Exception: {e.__cause__}')
        finally:
            if process_q.empty():
               exit(0)

    def start_processing(self, group_keys: List, batch_process_limit: int = None) -> h5.File.keys:
        """Start the data processing pipeline.
        :param group_keys is list of group names
        :param batch_process_limit is the maximum number of splits to create
        :returns h5.File keys"""
        try:
            """h5 file is closed after the initiation of the class; open it here"""
            h5_file = h5.File(self.__output_file, mode=self.__write_mode)
            paths, process_q, local_mpq = self.__process_schema_input(h5_file)
            for gkey in group_keys:
                path = [p for p in paths if p[0].endswith(gkey)]
            else:
                print("No schema file found - using defaults")

            output_groups = self.__schema_dict['files']['outgoing'][h5_file.filename.split('/')[-1]]['groups']
            print(f"output_groups: {output_groups}")

            if process_q.empty():
                self.__write_content_to_file(h5_file)
                """ Close the file to avoid locking """
                h5_file.close()
                self.__input_file_type(process_q, batch_process_limit)
                h5_file.flush()
                """ flush memory content to disk """
            elif self.__input_dict:
                content_list = [self.__input_dict]
                file_list: List = []
                file_group = 'root'
                process_q.put([file_group, content_list, file_list])
                self.__data_handler(file_group=file_group, file_list=file_list, process_q=process_q)

            """ Shutdown of Queue and Process """
            if process_q.empty():
               local_mpq.current_process.close()

        except Exception as e:
            local_mpq.current_process.close()
            print(f'start_processor Exception: {e.__cause__}')
        except BaseException as be:
            print(f'start_processor BaseException: {be.__cause__}')
            pass
        finally:
            h5_file = h5.File(self.__output_file, mode=self.__write_mode)
            h5_keys = h5_file.keys()
            """ Close H5 file again when finished """
            h5_file.close()
            return h5_keys

