import multiprocessing
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from fileWrangler import fileHandler
import numpy as np
import h5py as h5
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from zipfile import ZipFile
from typing import AnyStr, Dict, List, Tuple
from PIL import Image
import tifffile
import json
import csv
import random
import gzip



class H5FileCreator:

    def __init__(self, output_file: AnyStr, write_mode: AnyStr = 'a') -> None:
        self.__output_file = output_file
        self.__write_mode = write_mode

    def create_file(self) -> h5.File:
        __h5_file = h5.File(self.__output_file, mode=self.__write_mode, libver='latest', locking=True, driver='stdio')
        return __h5_file

class H5DataCreator:
    def __init__(self, output_file: AnyStr, input_file: AnyStr,
                 input_dict: Dict | np.ndarray = None, input_dict_label: AnyStr = None) -> None:
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
            self.__input_dict_label = input_dict_label
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

    def __process_json(self):
        pass

    def __process_csv(self):
        pass

    def __process_file_groups(self):
        pass

    def __create_file_group(self, group_name: AnyStr) -> Tuple:
        while group_name.endswith('/'):
            created = self.__h5_file.get(group_name)
            if not created:
                self.__h5_file.create_group(group_name, track_order=True)
                self.__write_content_to_file()
                return 0, 0

    def __convert_images(self, process_q: multiprocessing.Queue, file: AnyStr,
                         open_file: ZipFile | h5.File | gzip.GzipFile, content_list: List, processed_file_list: List[str]):
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        # Use multiprocessing and queues for large image lists
        try:
            if file.endswith(image_extensions):
                ds = file.split('/')[0]
                if file.endswith('tiff'):
                    img = tifffile.imread(file)
                    content_list.append([ds, img])
                    processed_file_list.append(file)
                else:
                    with open_file.open(file) as img_file:
                        with ThreadPoolExecutor(max_workers=6) as executor:
                            img = executor.map((np.array(Image.open())), img_file)
                            content_list.append([ds, img])
                            processed_file_list.append(file)
                process_q.put([content_list, processed_file_list])
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
                    process_q.put([content_list, processed_file_list])
                finally:
                # Ensure temp file is removed, even in case of failures.
                    if os.path.exists(temp_img):
                        os.remove(temp_img)
        except Exception as e:
            print(e)
            self.__logger.error(f"Error processing file {file}: {e}")

    def classify_inputs(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile) -> Tuple:
        """Classify and process input files content into structured data formats."""
        content_list: List = []
        processed_file_list: List[str] = []
        process_q = multiprocessing.Queue()
        local_process = multiprocessing.Process(target=self.__write_content_to_file(), args=(process_q,))
        local_process.start()
        # local_joined_process = local_process.join()
        # declare image file name extensions
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        try:
            if file.endswith('/'):
              f, g = self.__create_file_group(file)
              if f == 0 and g == 0:
                process_q.put([f, g])
                content_list.append(['Group', file])
                processed_file_list.append(file)
                return content_list, processed_file_list

            # Process JSON files here
            if file.endswith('json'):
                file_name = file.casefold()
                raw_content = open_file.read(file).decode('utf-8').splitlines()
                content = [row for row in raw_content]
                content_dict = json.loads(content[0])
                content_list.append([file_name, content_dict])
                line_count = len(content_list)
                processed_file_list.append(file_name + '-' + str(line_count))
                process_q.put([content_list, processed_file_list])
            # Process image files
            elif file.endswith(image_extensions):
                with ThreadPoolExecutor(max_workers=6) as executor:
                    status = executor.map(self.__convert_images, [process_q, file, open_file, content_list, processed_file_list])
                    for result in status:
                        print(result)
            # Process CSV files
            elif file.endswith('csv'):
                with open_file.open(file) as csv_file:
                    csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
                                            doublequote=True, quotechar='"')
                    content = [row for row in csv_reader]
                    content_list.append([file, content])
                    processed_file_list.append(file)
                    process_q.put([content_list, processed_file_list])
        except Exception as e:
            self.__logger.getLogger(__name__).error(f"Error processing file {file}: {e}")
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
                    with ThreadPoolExecutor(max_workers=6) as executor:
                        status = executor.map(self.__parse_data, [v])
                        for result in status:
                            value_list.extend(result)
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
        return value_list

    @property
    def input_processor(self) -> h5.File.keys:
        try:
          file_processor = fileHandler(self.__input_file)
          if file_processor.input_file_type() != 'unknown':
            content, files = file_processor.input_file_type()

          elif self.__input_dict:
            content_list = [self.__input_dict]
            file_list: List = []

          if content and files:
            for contents, files in content_list, file_list:
                if content == 'Group':
                    print("Group Created")
                    break

                for file_group, content_lines in contents:
                  if not self.__h5_file.get(file_group):
                    self.__h5_file.create_group(file_group, track_order=True)
                    self.__h5_file[file_group].attrs['file_name'] = file_group
                    self.__h5_file[file_group].attrs['content_list_length'] = len(content_list)
                    print("write file group attrs")
                    self.__write_content_to_file()
                  else:
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
                          with ThreadPoolExecutor(max_workers=6) as executor:
                            kv_list = executor.map(self.__parse_data, line)
                            kva = np.array(kv_list, dtype=np.dtype('S'))
                          idx = list_count
                          self.__h5_file.require_group(file_group).attrs[f'{idx}'] = kva
                          self.__write_content_to_file()
                          list_count += 1

                        elif isinstance(line, np.ndarray):
                            if self.__h5_file.get(f'{file_group}'):
                                self.__h5_file[f'{file_group}'].create_dataset(f'{file_group}', data=line,
                                                                                           compression='gzip',
                                                                                           chunks=True)
                                self.__write_content_to_file()

          elif self.__input_dict:
            with ThreadPoolExecutor(max_workers=6) as executor:
               value_list: List = executor.map(self.__parse_data,self.__input_dict)

            file_group = 'root'
            if not self.__h5_file.get(file_group):
              self.__h5_file.require_group(file_group)
              self.__h5_file[file_group].attrs['group_name'] = file_group
              self.__h5_file[file_group].attrs[self.__input_dict_label] = np.array(value_list, dtype=np.dtype('S'))
              self.__write_content_to_file()

        except Exception as e:
          self.__h5_file.close()
          self.__logger.error(f"Error processing input file: {e}")
          return e
        finally:
          self.__h5_file.close()
          return self.__h5_file.keys()
