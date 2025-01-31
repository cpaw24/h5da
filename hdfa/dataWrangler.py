import multiprocessing
import os
import uuid

from mpLocal import MpQLocal
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
    def __init__(self, output_file: AnyStr, write_mode: AnyStr = 'a') -> None:
        self.__output_file = output_file
        self.__write_mode = write_mode

    def create_file(self) -> h5.File:
        __h5_file = h5.File(self.__output_file, locking=False, mode=self.__write_mode, libver='latest', driver=None)
        __h5_file.clear()
        __h5_file.close()
        return __h5_file

class H5DataCreator:
    def __init__(self, output_file: h5.File, input_file: AnyStr,
                 input_dict: Dict | np.ndarray = None, write_mode: AnyStr = 'a', schema_file: AnyStr = None) -> None:
        """
        Initialize the H5DataCreator class.
        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz, .h5).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        """
        self.__input_file = input_file
        self.__output_file = output_file
        self.__input_dict = input_dict
        self.__write_mode = write_mode
        self.__schema_file = schema_file
        self.__h5_file = output_file

    def random_int_generator(self) -> str:
        random_int = random.randint(1, 1000000)
        return str(random_int)

    def __write_content_to_file(self) -> None:
        self.__h5_file.flush()

    @property
    def h5_file(self):
        return h5.File(self.__h5_file, mode=self.__write_mode, libver='latest', driver=None)

    def __create_dataset_from_dict(self, name: AnyStr, data: Dict, file_group: AnyStr, h5_file: h5.File) -> None:
        kv_list = self.__parse_data(input_dict=data)
        kva = np.array(kv_list, dtype=np.dtype('S'))
        h5_file.require_group(file_group).create_dataset(name, data=kva, compression='gzip')

    def create_dataset_from_input(self, data: List | Dict | np.ndarray, file_group: AnyStr, h5_file: h5.File) -> None:
        if isinstance(data, List):
           init_byte = data[0]
           if isinstance(init_byte, int):
              kva = np.array(data, dtype=np.dtype('I'))
           elif isinstance(init_byte, str):
              kva = np.array(data, dtype=np.dtype('S'))

              name = 'ds-' + str(uuid.uuid4())[:8]
              h5_file.require_group(file_group).create_dataset(name=name, data=kva, compression='gzip')
        elif isinstance(data, List):
           list_count = 0
           name = 'ds-' + str(uuid.uuid4())[:8]
           if isinstance(data[0], Dict):
              for l in data:
                 idx = list_count
                 name = name + '-' + str(idx)
                 self.__create_dataset_from_dict(name=name, data=l, file_group=file_group, h5_file=h5_file)
                 list_count += 1
        elif isinstance(data, Dict):
           name = 'ds-' + str(uuid.uuid4())[:8]
           self.__create_dataset_from_dict(name=name, data=data, file_group=file_group, h5_file=h5_file)
        elif isinstance(data, np.ndarray):
           name = 'ds-' + str(uuid.uuid4())[:8]
           h5_file.require_group(file_group).create_dataset(name=name, data=data, compression='gzip')

        return h5_file.flush()

    def __process_json(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                       content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]]:
        file_name = file.casefold()
        raw_content = open_file.read(file).decode('utf-8').splitlines()
        content = [row for row in raw_content]
        content = json.loads(content[0])
        content_list.append([file_name, content])
        line_count = len(content_list)
        processed_file_list.append(file_name + '-' + str(line_count))
        return content_list, processed_file_list

    def __process_csv(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]]:
        with open_file.open(file) as csv_file:
           csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
                                   doublequote=True, quotechar='"')
           content = [row for row in csv_reader]
           content_list.append([file, content])
           processed_file_list.append(file)
           return content_list, processed_file_list

    def __process_svg(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]]:
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
            if os.path.exists(temp_img):
                os.remove(temp_img)
        finally:
            # Ensure temp file is removed.
            if os.path.exists(temp_img):
                os.remove(temp_img)

    def create_file_group(self, group_name: AnyStr, h5_file: h5.File = None, content_size: int = 0):
        created = h5_file.get(group_name)
        if not created:
           h5_file.create_group(group_name, track_order=True)
           h5_file[group_name].attrs['file_name'] = h5_file.name
           h5_file[group_name].attrs['content_size'] = content_size
           print("write file group attrs")

    def get_file_group(self, group_name: AnyStr, h5_file: h5.File):
        return h5_file.get(group_name)

    def __write_schema(self, path: List,
                     content_list: List, processed_file_list: List[str]):
        flg: AnyStr = ''
        if not content_list: content_list: List = []
        if not processed_file_list: processed_file_list: List = []
        for l in path:
            if isinstance(l, List):
               for i in l:
                  flg = i
            else:
               flg = l

               self.create_file_group(flg)

            content_list.append(['Group', flg])
            processed_file_list.append(flg)
            return content_list, processed_file_list


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
                     img = np.array(Image.open(img_file))
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
        except Exception as e:
            print(e)

    def __classify_inputs(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                          process_q: multiprocessing.Queue) -> Tuple[multiprocessing.Queue, List, List[str]]:
        """Classify and process input files content into structured data formats."""
        content_list: List = []
        processed_file_list: List[str] = []
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        try:
            if not file.endswith('/'):
                # Process JSON files
                if file.endswith('json') or file.endswith('jsonl'):
                   content_list, processed_file_list = self.__process_json(file, open_file,
                                                                           content_list, processed_file_list)
                   process_q.put([content_list, processed_file_list])
                # Process image files
                elif file.endswith(image_extensions):
                   self.__convert_images(process_q, file, open_file,
                                   content_list, processed_file_list)
                # Process CSV files
                elif file.endswith('csv'):
                    content_list, processed_file_list = self.__process_csv(file, open_file,
                                                                           content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list])

        except Exception as e:
            print(e)

        return process_q, content_list, processed_file_list

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

    def __open_zip(self, process_q: multiprocessing.Queue) -> Tuple[multiprocessing.Queue, List, List[str]]:
       zip = ZipFile(self.__input_file, 'r')
       file_list = self.__file_list()
       for file in file_list[0:500]:
          if not file.endswith('/'):
             process_q, content_list, processed_file_list = self.__classify_inputs(file, zip, process_q)

       return process_q, content_list, processed_file_list

    def __open_h5(self, process_q: multiprocessing.Queue) -> Tuple[multiprocessing.Queue, List, List[str]]:
       h5file = self.__input_file.h5.open().read().decode('utf-8')
       process_q, content_list, processed_file_list = self.__classify_inputs(h5file, h5file, process_q)
       return process_q, content_list, processed_file_list

    def __open_gzip(self, process_q: multiprocessing.Queue) -> Tuple[multiprocessing.Queue, List, List[str]]:
       gzfile = gzip.open(self.__input_file, 'r', encoding='utf-8')
       file_list = self.__file_list()
       for file in file_list:
          process_q, content_list, processed_file_list = self.__classify_inputs(file, gzfile, process_q)
          return process_q, content_list, processed_file_list

    def __get_schemas(self) -> Tuple[List, List]:
        in_path_list = []
        out_path_list = []
        if os.path.exists(self.__schema_file):
            files = ['incoming', 'outgoing']
            schema = json.load(open(self.__schema_file))
            schema_file = schema.get('files')
            for file in files:
               file_path = schema_file.get(file)
               if file == 'incoming':
                  file_name = file_path.get(self.__input_file.split('/')[-1])
               elif file == 'outgoing':
                  file_name = file_path.get(self.__output_file.split('/')[-1])

                  level = ''
                  level_size = len(file_name) + 1
                  for i in range(level_size):
                     if i >= 1:
                        levels = file_name[f"L{i}"]
                        if not isinstance(levels, List):
                           level = level + levels
                        elif isinstance(levels, List):
                           current_level = level
                           for l in range(len(levels)):
                              level = current_level + levels[l]
                              if file == 'incoming':
                                 in_path_list.append([level])
                              elif file == 'outgoing':
                                 out_path_list.append([level])
            return in_path_list, out_path_list
        else:
            print("No schema file found - return empty lists")
            return in_path_list, out_path_list

    def input_file_type(self, process_q: multiprocessing.Queue) -> str | Tuple[multiprocessing.Queue, List, List]:
       if self.__input_file.endswith('zip' or 'z'):
           process_q, content_list, processed_file_list = self.__open_zip(process_q)
           return process_q, content_list, processed_file_list
       elif self.__input_file.endswith('h5' or 'hdf5'):
           process_q, content_list, processed_file_list = self.__open_h5(process_q)
           return process_q, content_list, processed_file_list
       elif self.__input_file.endswith('gz' or 'gzip'):
           process_q, content_list, processed_file_list = self.__open_gzip(process_q)
           return process_q, content_list, processed_file_list
       else:
           return multiprocessing.Queue(), ['unknown'], ['unknown']

    def process_content_list(self, content: List, file_group: AnyStr, h5_file: h5.File = None):
        for file, line in content:
            self.create_dataset_from_input(data=line, file_group=file_group, h5_file=h5_file)
            self.__write_content_to_file()

    def start_mp(self):
        # Use multiprocessing and queues for large image lists
        local_mpq = MpQLocal()
        process_q = local_mpq.get_queue()
        return process_q

    def process_schema_file(self, schema_file: AnyStr,  h5_file: h5.File, content_list: List = None,
                            processed_file_list: List = None) -> None | List | str:
        # get schemas
        if schema_file:
            h5_file.require_group('root')
            h5_file.attrs['schema_file'] = schema_file
            in_path, out_path = self.__get_schemas()
            # check for output schema
            if len(out_path) > 0:
                self.__write_schema(out_path, content_list, processed_file_list)
                return out_path
            # duplicate the input paths in output file if no output schema present
            elif len(in_path) > 0 and len(out_path) == 0:
                self.__write_schema(in_path, content_list, processed_file_list)
                return in_path
            else:
                return "No schema to process - using defaults"

class InputProcessor:
    def __init__(self, input_dict: Dict, input_file: AnyStr, output_file: AnyStr, schema_file: AnyStr = None):
        self.__input_file = input_file
        self.__output_file = output_file
        self.__input_dict = input_dict
        self.__schema_file = schema_file
        self.__content_list = None
        self.__processed_file_list = None
        self.__file_group = 'root'

    def file_processor(self) -> h5.File.keys:
      try:
         image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
         if os.path.exists(self.__schema_file):
            dc = H5DataCreator(input_file=self.__input_file, output_file=self.__output_file,
                                schema_file=self.__schema_file)
            h5_file = dc.h5_file
            h5_file.atomic
            paths = dc.process_schema_file(schema_file=self.__schema_file, h5_file=h5_file)
            # Use multiprocessing and queues for large image lists
            process_q = dc.start_mp()
            for file in paths:
                if file[0].endswith('images/'):
                    images = file[0]
                if file[0].endswith('styles/'):
                    style = file[0]
         else:
             print("No schema file found - using defaults")

         if dc.input_file_type(process_q) != [multiprocessing.Queue, 'unknown', 'unknown']:
            process_q, content_list, processed_file_list = dc.input_file_type(process_q)
         elif self.__input_dict:
            content_list = [self.__input_dict]
            file_list: List = []
            file_group = 'root'

         while not process_q.empty():
            for data in process_q.get_nowait():
               if isinstance(data, List):
                  if len(data[0]) == 2:
                     file_group, content_list = data[0]
                     if isinstance(content_list, np.ndarray):
                        dc.create_dataset_from_input(data=content_list, file_group=images, h5_file=h5_file)
                  elif str(data[0]).endswith(image_extensions):
                     ds_file_name = data[0]
                     h5_file.require_group(images).attrs[ds_file_name] = True
                  else:
                     for contents, file in content_list, file_list:
                        for file_group, content_lines in contents:
                           if not dc.get_file_group(file_group, h5_file):
                              dc.create_file_group(file_group, h5_file.filename, len(content_list))
                              print("write file group attrs")
                           elif not file_list:
                              file_group = 'root'
                              if not dc.get_file_group(file_group, h5_file):
                                 dc.create_file_group(file_group, h5_file, len(content_list))
                                 print("write root attrs")

                        dc.process_content_list(content=content_lines, file_group=file_group)
         else:
            dc.process_content_list(content=content_list, file_group=file_group)

      except Exception as e:
         h5_file.flush()
         h5_file.close()
         return e
      except BaseException as be:
         print(be)
         pass
      finally:
         h5_file.flush()
         h5_file.close()
         return h5_file.keys()



