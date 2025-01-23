import multiprocessing
import os
import logging
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, output_file: AnyStr, input_file: AnyStr, schema_file: AnyStr, input_dict: Dict | np.ndarray = None) -> None:
        """
        Initialize the H5DataCreator class.
        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz, .h5).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        """
        self.__input_file = input_file
        self.__output_file = output_file
        self.__input_dict = input_dict
        self.__schema_file = schema_file

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

    def __create_dataset_from_dict(self, name: AnyStr, data: Dict, file_group: AnyStr) -> None:
        kv_list = self.__parse_data(input_dict=data)
        kva = np.array(kv_list, dtype=np.dtype('S'))
        self.__h5_file.require_group(file_group).create_dataset(name, data=kva,
                                                                   compression='gzip',
                                                                   chunks=True)
        self.__write_content_to_file()

    def __create_dataset_from_input(self, data: List | Dict | np.ndarray, file_group: AnyStr, file: AnyStr) -> None:
        if isinstance(data, List):
           list_count = 0
           if isinstance(data[0], Dict):
              for l in data:
                 idx = list_count
                 self.__create_dataset_from_dict(name=f'{idx}', data=l, file_group=file_group)
                 list_count += 1
        elif isinstance(data, Dict):
           self.__create_dataset_from_dict(name=file, data=data, file_group=file_group)
        elif isinstance(data, np.ndarray):
           self.__h5_file.require_group(file_group).create_dataset(file, data=data, compression='gzip', chunks=True)
        elif isinstance(data, (int, str)):
           if isinstance(data, int):
              kva = np.array(data, dtype=np.dtype('I'))
           elif isinstance(data, str):
              kva = np.array(data, dtype=np.dtype('S'))

              self.__h5_file.require_group(file_group).create_dataset(file, data=kva, compression='gzip', chunks=True)

    def __process_json(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                       content_list: List, processed_file_list: List[str]):
        file_name = file.casefold()
        raw_content = open_file.read(file).decode('utf-8').splitlines()
        content = [row for row in raw_content]
        content = json.loads(content[0])
        content_list.append([file_name, content])
        line_count = len(content_list)
        processed_file_list.append(file_name + '-' + str(line_count))
        return content_list, processed_file_list

    def __process_csv(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]):
        with open_file.open(file) as csv_file:
           csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
                                   doublequote=True, quotechar='"')
           content = [row for row in csv_reader]
           content_list.append([file, content])
           processed_file_list.append(file)
           return content_list, processed_file_list

    def __process_svg(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]):
        ds = file.split('/')[0]
        temp_img = f"{self.random_int_generator()}_temp.png"
        try:
           drawing = svg2rlg(open_file.open(file))
           renderPM.drawToFile(drawing, temp_img, fmt='PNG')
           img = np.array(Image.open(temp_img))
           content_list.append([ds, img])
           processed_file_list.append(file)
           return content_list, processed_file_list

        except Exception as e:
            print(e)
            self.__logger.error(f"Error processing file {file}: {e}")
            if os.path.exists(temp_img):
                os.remove(temp_img)
        finally:
            # Ensure temp file is removed.
            if os.path.exists(temp_img):
                os.remove(temp_img)

    def __create_file_group(self, group_name: AnyStr, file_name: AnyStr) -> Tuple:
        created = self.__h5_file.get(group_name)
        if not created:
           self.__h5_file.create_group(group_name, track_order=True)
           self.__h5_file[group_name].attrs['file_name'] = file_name
           print("write file group attrs")
           self.__write_content_to_file()
           return 0, 0

    def __convert_images(self,  process_q: multiprocessing.Queue, file: AnyStr,
                         open_file: ZipFile | h5.File | gzip.GzipFile, content_list: List, processed_file_list: List[str]):
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        try:
           if file.endswith(image_extensions):
              ds = file.split('/')[0]
              if file.endswith('tiff'):
                 img = tifffile.imread(file)
                 content_list.append([ds, img])
                 processed_file_list.append(file)
                 process_q.put([content_list, processed_file_list])
              else:
                  with open_file.open(file) as img_file:
                     with ThreadPoolExecutor(max_workers=6) as executor:
                        img = executor.map((np.array(Image.open())), img_file)
                        content_list.append([ds, img])
                        processed_file_list.append(file)
                     process_q.put([content_list, processed_file_list])
            # Process SVG files (Convert to numpy array via PNG generation)
           elif file.endswith('svg'):
              try:
                 content_list, processed_file_list = self.__process_svg(file, open_file,
                                                                        content_list, processed_file_list)
                 process_q.put([content_list, processed_file_list])
              except Exception as e:
                 print(e)
                 self.__logger.error(f"Error processing file {file}: {e}")

        except Exception as e:
            print(e)
            self.__logger.error(f"Error processing file {file}: {e}")

    def __classify_inputs(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile) -> Tuple:
        """Classify and process input files content into structured data formats."""
        content_list: List = []
        processed_file_list: List[str] = []
        # Use multiprocessing and queues for large image lists
        process_q = multiprocessing.Queue()
        local_process = multiprocessing.Process(target=self.__write_content_to_file(), args=(process_q,))
        local_process.start()
        # local_joined_process = local_process.join()
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        try:
            # get schemas
            in_path, out_path = self.__get_schemas()
            # check for output schema
            if len(out_path) > 0:
               for l in out_path:
                  while l.endswith('/'):
                     f, g = self.__create_file_group(l, self.__output_file)
                     process_q.put([f, g])
                     content_list.append(['Group', l])
                     processed_file_list.append(l)
               return content_list, processed_file_list
            # duplicate the input paths in output file if no output schema present
            elif len(in_path) > 0 and len(out_path) == 0:
               for l in in_path:
                  while l.endswith('/'):
                     f, g = self.__create_file_group(l, self.__input_file)
                     process_q.put([f, g])
                     content_list.append(['Group', l])
                     processed_file_list.append(l)
               return content_list, processed_file_list

            # Process JSON files
            if file.endswith('json') or file.endswith('jsonl'):
               content_list, processed_file_list = self.__process_json(file, open_file,
                                                                       content_list, processed_file_list)
               process_q.put([content_list, processed_file_list])
            # Process image files
            elif file.endswith(image_extensions):
                with ThreadPoolExecutor(max_workers=6) as executor:
                   executor.map(self.__convert_images,
                               [process_q, file, open_file,
                               content_list, processed_file_list])
            # Process CSV files
            elif file.endswith('csv'):
                content_list, processed_file_list = self.__process_csv(file, open_file,
                                                                       content_list, processed_file_list)
                process_q.put([content_list, processed_file_list])

        except Exception as e:
            self.__logger.error(f"Error processing file {file}: {e}")
            print(e)

        return process_q, local_process

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
                 value_list.extend([self.__parse_data(v)])
              elif isinstance(v, (List, np.ndarray)):
                 # Check if the list or array contains integers
                 if all(isinstance(i, int) for i in v):
                    # Ensure v is converted to a numpy array only when needed
                    value_list.append([k, np.ndarray(v)])
                 else:
                    # Add raw lists if not integers
                    value_list.append([k, v])
              elif isinstance(v, Tuple):
                 _a, _b, _c = v
                 value_list.append([k, [_a, _b, _c]])
              elif isinstance(v, (int, str)):
                 # Add primitive types (e.g., strings, numbers)
                 value_list.append([k, v])
        elif isinstance(input_dict, np.ndarray):
           # If the input is a numpy array, append it directly
           value_list.append([input_dict])
        elif isinstance(input_dict, Tuple):
           _a, _b, _c = input_dict
           value_list.append([_c,(_a, _b)])
        return value_list

    def __file_list(self) -> List:
       if self.__input_file.endswith('zip' or 'z'):
          with ZipFile(self.__input_file, 'r') as zip:
             file_list = zip.namelist()
             return file_list
       elif self.__input_file.endswith('gz' or 'gzip'):
          with gzip.open(self.__input_file, 'rb') as gzf:
             file_list = gzip.GzipFile(fileobj=gzf.fileobj.__dict__.get('namelist'))
             return file_list

    def __open_zip(self) -> Tuple[List, List]:
       zip = ZipFile(self.__input_file, 'r')
       file_list = self.__file_list()
       for file in file_list:
          process_q, local_process = self.__classify_inputs(file, zip)
          return process_q, local_process

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

    def __get_schemas(self) -> Tuple[List, List]:
        in_path_list = []
        out_path_list = []
        if os.path.exists(self.__schema_file):
            schema = json.loads(self.__schema_file)
            in_schema = schema['incoming_file_schema']
            in_file_name = in_schema[self.__input_file]
            in_levels: List = []
            in_level_size = len(in_file_name)
            for i in range(in_level_size):
                in_levels.append([in_file_name.get(f"L{i}"), in_file_name])

            out_schema = schema['outgoing_file_schema']
            out_file_name = out_schema[self.__output_file]
            out_levels: List = []
            out_level_size = len(out_file_name)
            for o in range(out_level_size):
               out_levels.append([out_file_name.get(f"L{o}"), out_file_name])

            return in_path_list, out_path_list
        else:
            return in_path_list, out_path_list

    def __input_file_type(self) -> str | Tuple[List, List]:
       if self.__input_file.endswith('zip' or 'z'):
           process_q, local_process = self.__open_zip()
           return process_q, local_process
       elif self.__input_file.endswith('h5' or 'hdf5'):
           process_q, local_process = self.__open_h5()
           return process_q, local_process
       elif self.__input_file.endswith('gz' or 'gzip'):
           process_q, local_process = self.__open_gzip()
           return process_q, local_process
       else:
           return ['unknown'], ['unknown']

    def input_processor(self) -> h5.File.keys:
      try:
         if self.__input_file_type() != 'unknown':
            process_q, local_process = self.__input_file_type()
         elif self.__input_dict:
            content_list = [self.__input_dict]
            file_list: List = []

         if process_q and local_process:
           for data in local_process.get(process_q):
              content_list, file_list = data
              for contents, file in content_list, file_list:
                 for file_group, content_lines in contents:
                    if not self.__h5_file.get(file_group):
                       self.__create_file_group(file_group, file)
                       self.__h5_file[file_group].attrs['content_list_length'] = len(content_list)
                       print("write file group attrs")
                       self.__write_content_to_file()
                    elif not file_list:
                       file_group = 'root'
                       if not self.__h5_file.get(file_group):
                          self.__create_file_group(file_group, file)
                          self.__h5_file[file_group].attrs['content_list_length'] = len(content_list)
                          print("write root attrs")
                          self.__write_content_to_file()

                    for file, line in content_list:
                       self.__create_dataset_from_input(data=line, file_group=file_group, file=file)
                       self.__write_content_to_file()

      except Exception as e:
         self.__h5_file.flush()
         self.__h5_file.close()
         self.__logger.error(f"Error processing input file: {e}")
         return e
      finally:
         self.__h5_file.close()
         return self.__h5_file.keys()



