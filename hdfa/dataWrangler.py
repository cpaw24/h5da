import gc
import math
import multiprocessing
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from hdfa.mpLocal import MpQLocal
import h5py as h5
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from zipfile import ZipFile
from typing import Any, AnyStr, Dict, Generator, List, Tuple
import numpy as np
from PIL import Image
import tifffile
import json
import csv
import random
import gzip

class DataProcessor:
    def __init__(self, output_file: AnyStr, input_file: AnyStr,
                 input_dict: Dict | np.ndarray = None, write_mode: AnyStr = 'a', schema_file: AnyStr = None) -> None:
        """
        Initialize the H5DataCreator class.
        :param output_file: Path to the output HDF5 file.
        :param input_file: Path to the input file (e.g., .zip, .gz, .h5).
        :param input_dict: Optional dictionary or ndarray to process and store in the HDF5 file.
        """
        self.__input_file = input_file
        self.__input_dict = input_dict
        self.__write_mode = write_mode
        self.__schema_file = schema_file
        self.__output_file = output_file
        self.__h5_file = self.__create_file()

    @staticmethod
    def random_int_generator() -> str:
        random_int = random.randint(1, 1000000)
        return str(random_int)

    def __create_file(self) -> h5.File:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with h5.File(self.__output_file, mode='w', libver='latest', driver='core', backing_store=True,
                     fs_persist=True, fs_strategy='page', fs_page_size=65536, page_buf_size=655360) as __h5_file:
           return __h5_file

    @staticmethod
    def __write_content_to_file(h5_file: h5.File) -> None:
        h5_file.clear()
        h5_file.flush()

    @staticmethod
    def __update_file_group(h5_file: h5.File, file_group: AnyStr, attr: int | AnyStr, attr_value: AnyStr | int) -> None:
        h5_file.get(file_group).attrs[f'{attr}'] = attr_value
        return h5_file.flush()

    def __create_dataset_from_dict(self, h5_file: h5.File, name: AnyStr, data: Dict, file_group: AnyStr) -> None:
        kv_list = self.__parse_data(input_dict=data)
        kva = np.array(kv_list, dtype=np.dtype('S'))
        h5_file.require_group(file_group).create_dataset(name, data=kva, compression='gzip')

    def __create_dataset_from_input(self, h5_file: h5.File, data: List | Dict | np.ndarray, file_group: AnyStr, file: AnyStr) -> None:
        h5_file.flush()
        if isinstance(data, List):
            init_byte = data[0]
            if isinstance(init_byte, int):
                kva = np.array(data, dtype=np.dtype('I'))
            elif isinstance(init_byte, str):
                kva = np.array(data, dtype=np.dtype('S'))
                name = 'ds-' + str(uuid.uuid4())[:8]
                h5_file.require_group(file_group).create_dataset(name=name, data=kva, compression='gzip')
                self.__update_file_group(h5_file=h5_file, file_group=file_group, attr=file, attr_value=True)
        elif isinstance(data, List):
            list_count = 0
            name = 'ds-' + str(uuid.uuid4())[:8]
            if isinstance(data[0], Dict):
                for l in data:
                    idx = list_count
                    name = name + '-' + str(idx)
                    self.__create_dataset_from_dict(name=name, data=l, file_group=file_group)
                    self.__update_file_group(h5_file=h5_file, file_group=file_group, attr=file, attr_value=True)
                    list_count += 1
        elif isinstance(data, Dict):
            name = 'ds-' + str(uuid.uuid4())[:8]
            self.__create_dataset_from_dict(name=name, data=data, file_group=file_group)
            self.__update_file_group(h5_file=h5_file, file_group=file_group, attr=file, attr_value=True)
        elif isinstance(data, np.ndarray):
            h5_file.flush()
            name = 'ds-' + str(uuid.uuid4())[:8]
            h5_file.require_group(file_group).create_dataset(name=name, data=data, compression='gzip')
            self.__update_file_group(h5_file=h5_file, file_group=file_group, attr=file, attr_value=True)

        return h5_file.flush()

    @staticmethod
    def __process_json(file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                       content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]]:
        file_name = file.casefold()
        raw_content = open_file.read(file).decode('utf-8').splitlines()
        content = [row for row in raw_content]
        content = json.loads(content[0])
        line_count = len(content_list)
        content_list.append([file_name, content, line_count])
        processed_file_list.append(file_name + '-' + str(line_count))
        return content_list, processed_file_list

    @staticmethod
    def __process_csv(file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]]:
        with open_file.open(file) as csv_file:
            csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
                                    doublequote=True, quotechar='"')
            content = [row for row in csv_reader]
            content_size = len(content)
            content_list.append([file, content, content_size])
            processed_file_list.append(file)
            return content_list, processed_file_list

    def __process_svg(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                      content_list: List, processed_file_list: List[str]) -> [List, List]:
        ds = file.split('/')[0]
        temp_img = f"{self.random_int_generator()}_temp.png"
        try:
            drawing = svg2rlg(open_file.open(file))
            renderPM.drawToFile(drawing, temp_img, fmt='PNG')
            img = np.array(Image.open(temp_img))
            content_list.append([ds, img, file])
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

    @staticmethod
    def __process_video(file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                        content_list: List, processed_file_list: List[str]) -> List[List | List[str]]:
        """Converts mp4/mp3 video to Numpy array"""
        import cv2
        ds = file.split('/')[0]
        frames = []

        with open_file.open(file) as video:
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

    def __create_file_group(self, h5_file: h5.File, group_name: AnyStr, content_size: int = 0):
        created = h5_file.get(group_name)
        if not created:
            h5_file.create_group(group_name, track_order=True)
            h5_file[group_name].attrs['file_name'] = h5_file.name
            h5_file[group_name].attrs['schema_file'] = self.__schema_file
            h5_file[group_name].attrs['content_size'] = content_size
            print("write file group attrs")
            return h5_file.clear()

    @staticmethod
    def __get_file_group(h5_file: h5.File, group_name: AnyStr):
        return h5_file.get(group_name)

    def __write_schema(self, h5_file: h5.File, path: List,
                       content_list: List, processed_file_list: List[str]):
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

            self.__create_file_group(h5_file, flg, content_length)

            content_list.append(['Group', flg])
            processed_file_list.append(flg)
        return content_list, processed_file_list

    def __convert_images(self, process_q: multiprocessing.Queue, file: AnyStr,
                         open_file: ZipFile | h5.File | gzip.GzipFile, content_list: List,
                         processed_file_list: List[str]):
        try:
            ds = file.split('/')[0]
            # Process SVG files (Convert to numpy array via PNG generation)
            if file.endswith('svg'):
               try:
                  content_list, processed_file_list = self.__process_svg(file, open_file,
                                                                       content_list, processed_file_list)
                  process_q.put(content_list)
               except Exception as e:
                  print(e)

            if file.endswith('tiff'):
                img = tifffile.imread(file)
                content_list.append([ds, img, file])
                process_q.put(content_list)
            else:
                with open_file.open(file) as img_file:
                    img = np.array(Image.open(img_file))
                    content_list.append([ds, img, file])
                    process_q.put(content_list)
        except Exception as e:
           print(e)

    def __classify_inputs(self, file: AnyStr, open_file: ZipFile | h5.File | gzip.GzipFile,
                          process_q: multiprocessing.Queue) -> None:
        """Classify and process input files content into structured data formats."""
        content_list: List = []
        processed_file_list: List[str] = []
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'blp', 'bufr', 'cur', 'dcx', 'dib', 'eps', 'fits',
                            'gif', 'jfif', 'jpe', 'ico', 'icns', 'j2c', 'j2k', 'jp2', 'jpc', 'jpx', 'pcx', 'pxr',
                            'pgm', 'ppm', 'sgi', 'svg')
        video_extensions = ('mpeg', 'mp4', 'mp3')
        try:
            if not file.endswith('/'):
                # Process JSON files
                if file.endswith('json') or file.endswith('jsonl'):
                    content_list, processed_file_list = self.__process_json(file, open_file,
                                                                            content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])
                # Process image files
                elif file.endswith(image_extensions):
                    self.__convert_images(process_q, file, open_file,
                                          content_list, processed_file_list)
                # Process CSV files
                elif file.endswith('csv'):
                    content_list, processed_file_list = self.__process_csv(file, open_file,
                                                                           content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])
                # Process video files
                elif file.endswith(video_extensions):
                    content_list, processed_file_list = self.__process_video(file, open_file,
                                                                             content_list, processed_file_list)
                    process_q.put([content_list, processed_file_list, file])

        except Exception as e:
            print(e)

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
            value_list.append([_c, (_a, _b)])
        return value_list

    def __process_content_list(self, content: List, file_group: AnyStr, h5_file: h5.File = None):
        for file, line in content:
            self.__create_dataset_from_input(h5_file=h5_file, data=line, file_group=file_group, file=h5_file.filename)
            self.__write_content_to_file(h5_file=h5_file)

    @staticmethod
    def __start_mp():
        # Use multiprocessing and queues for large image lists
        local_mpq = MpQLocal()
        process_q = local_mpq.get_queue()
        return process_q, local_mpq

    def __file_list(self) -> List:
        if self.__input_file.endswith('zip' or 'z'):
            with ZipFile(self.__input_file, 'r') as zip:
                file_list = zip.namelist()
                return file_list
        elif self.__input_file.endswith('gz' or 'gzip'):
            with gzip.open(self.__input_file, 'rb') as gzf:
                file_list = gzip.GzipFile(fileobj=gzf.fileobj.__dict__.get('namelist'))
                return file_list

    def __size_batching(self, file_list: List, batch_size: int):
        batches = math.ceil(len(file_list) / batch_size)
        batch_list: List = []
        for i in range(batches):
            start_batch = i * batch_size
            end_batch = start_batch + batch_size - 1
            batch_list.append(file_list[start_batch:end_batch])
        return batch_list

    def __input_file_type(self, process_q: multiprocessing.Queue, batch_process_limit: int) -> str | None:
        if self.__input_file.endswith('zip' or 'z'):
            zip = ZipFile(self.__input_file, 'r')
            file_list = self.__file_list()
            batch_chunk_size = 1000
            batch_list = self.__size_batching(file_list, batch_chunk_size)
            b_counter = 0
            print(f"Processing {len(file_list)} files in batches of 1000")
            for batch in batch_list:
               for file in batch:
                  if not file.endswith('/'):
                     self.__classify_inputs(file, zip, process_q)
                  else:
                     print(f"Skipping directory: {file}")
                     continue
               gc.collect()
               b_counter += 1
               # Process data on Queue every five batches to free memory
               if b_counter == batch_process_limit:
                  self.__data_handler(process_q, file_group=file.split('/')[0], file_list=file_list)
                  b_counter = 0

        elif self.__input_file.endswith('h5' or 'hdf5'):
           h5file = self.__input_file.h5.open().read().decode('utf-8')
           self.__classify_inputs(h5file, h5file, process_q)
        elif self.__input_file.endswith('gz' or 'gzip'):
            gzfile = gzip.open(self.__input_file, 'r', encoding='utf-8')
            file_list = self.__file_list()
            for file in file_list:
                self.__classify_inputs(file, gzfile, process_q)
        else:
            return 'unknown'

    def __process_schema_file(self, h5_file: h5.File, content_list: List = None,
                            processed_file_list: List = None) -> None | List | str:
        try:
            if not content_list: content_list: List = []
            if not processed_file_list: processed_file_list: List = []
            # get schemas
            h5_file.require_group('root')
            in_path, out_path = self.__get_schemas()
            # check for output schema
            if len(out_path) > 0:
                self.__write_schema(h5_file=h5_file, path=out_path,
                                    content_list=content_list, processed_file_list=processed_file_list)
                return out_path
            # duplicate the input paths in output file if no output schema present
            elif len(in_path) > 0 and len(out_path) == 0:
                self.__write_schema(h5_file=h5_file, path=in_path,
                                    content_list=content_list, processed_file_list=processed_file_list)
                return in_path
            else:
                return "No schema to process - using defaults"
        except Exception as e:
            print(e)

    def __get_schemas(self) -> Tuple[List, List]:
        in_path_list = []
        out_path_list = []
        schema_file = self.__schema_file
        files = ['incoming', 'outgoing']
        schema = json.load(open(schema_file, 'r'))
        schema_file = schema.get('files')
        for file in files:
            file_path = schema_file.get(file)
            if file == 'incoming':
                file_name = file_path.get(self.__input_file.split('/')[-1])
            elif file == 'outgoing':
                 file_name = file_path.get(self.__h5_file.name.split('/')[-1])

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

    def __process_schema_input(self, h5_file: h5.File, content_list: List = None):
        if os.path.exists(self.__schema_file):
            paths = self.__process_schema_file(h5_file, content_list)
            # Use multiprocessing and queues for large image lists
            process_q, local_mpq = self.__start_mp()
            process_q.get()
            return paths, process_q, local_mpq
        else:
            print("No schema file found - using defaults")

    def __data_handler(self, process_q: multiprocessing.Queue, file_group: AnyStr, file_list: AnyStr):
        while not process_q.empty():
            for data in process_q.get():
                if isinstance(data, List):
                    file_group, content_list, file = data
                    h5_file = h5.File(self.__output_file, mode=self.__write_mode)
                    if isinstance(content_list, np.ndarray):
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            executor.submit(self.__create_dataset_from_input, h5_file=h5_file, data=content_list,
                                            file_group=file_group, file=file)
                            print(f"write ds - {file}")
                        # self.__create_dataset_from_input(h5_file=h5_file, data=content_list,
                        #                                 file_group=file_group, file=file)
                        print(f"write ds - {file}")
                    elif isinstance(data, Dict):
                        self.__create_dataset_from_input(h5_file=h5_file, data=data, file_group=file_group, file=file)
                        print(f"write ds - {file}")
                    else:
                        for contents, file in content_list, file_list:
                            for file_group, content_lines in contents:
                                if not self.__get_file_group(h5_file=h5_file, group_name=file_group):
                                    self.__create_file_group(file_group, len(content_list))
                                    print("write file group attrs")
                                elif not file_list:
                                    file_group = 'root'
                                    if not self.__get_file_group(h5_file=h5_file, group_name=file_group):
                                        self.__create_file_group(h5_file=h5_file, group_name=file_group,
                                                                 content_size=len(content_list))
                                        print("write root attrs")

                                    self.__process_content_list(content=content_lines, file_group=file_group)
                else:
                    self.__process_content_list(content=content_list, file_group=file_group)

    def file_processor(self, group_keys: List, batch_process_limit: int) -> h5.File.keys:
        try:
            h5_file = h5.File(self.__output_file, mode=self.__write_mode)
            paths, process_q, local_mpq = self.__process_schema_input(h5_file)
            for gkey in group_keys:
                path = [p for p in paths if p[0].endswith(gkey)]
            else:
                print("No schema file found - using defaults")

            if process_q.empty():
                self.__write_content_to_file(h5_file)
                # Close the file to avoid locking
                h5_file.close()
                self.__input_file_type(process_q, batch_process_limit)
                h5_file.flush()
                h5_file.close()
                # Completeion
            elif self.__input_dict:
                content_list = [self.__input_dict]
                file_list: List = []
                file_group = 'root'
                process_q.put([file_group, content_list, file_list])
                self.__data_handler(file_group=file_group, file_list=file_list, process_q=process_q)

            # Shutdown of Queue and Process
            if process_q.empty():
               local_mpq.current_process.close()

        except Exception as e:
            local_mpq.current_process.close()
            return e
        except BaseException as be:
            print(be)
            pass
        finally:
            h5_file = h5.File(self.__output_file, mode=self.__write_mode)
            h5_keys = h5_file.keys()
            return h5_keys

