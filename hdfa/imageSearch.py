import os
import logging
import h5rdmtoolbox as h5tbx
import lmdb
import json
from click import open_file
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from dataWrangler import DataProcessor
from typing import AnyStr, List, Tuple
import numpy as np
from PIL import Image
import tifffile


class SearchImages:
    """Not Implemented."""
    def __init__(self, input_file: AnyStr, searched_files: List, use_path: AnyStr = os.getcwd(), config_file: AnyStr = None):
        self.__input_file = input_file
        self.__use_path = use_path
        self.__searched_files = searched_files
        self.__config_file = json.load(open(config_file, 'r'))
        self.__initialize()

    def __initialize(self):
        self.__logger = logging.getLogger(__name__)
        self.__input_file_size = os.path.getsize(self.__input_file)
        self.H5File = h5tbx.File(self.__input_file, 'r')
        """ Create LMDB database for image index. Map size is 1024MB for small files and 1GB for large files."""
        if self.__input_file_size <= 100000000:
            self.idx_db = lmdb.open(os.path.join(self.__use_path, 'image_index.lmdb'), map_size=(1024 * 1024 * 102.4))
        elif self.__input_file_size > 100000000:
            self.idx_db = lmdb.open(os.path.join(self.__use_path, 'image_index.lmdb'), map_size=(1024 * 1024 * 1024))

    def __store_index(self, source_data: Tuple) -> bool | None:
        try:
            with self.idx_db.begin(write=True).cursor() as txn:
                i, k, v = source_data
                txn.put(bytes(k, v))
                txn.commit()
                return True
        except Exception as e:
            print(e)
            txn.rollback()
            self.idx_db.close()
            return False

    def __raise_index_error(self) -> None:
        raise Exception("Error storing index")

    def __content_index(self):
        image_extensions = self.__config_file.get('image_extensions')
        __files = self.__searched_files
        for __file in __files:
           if __file.endswith(image_extensions):
              try:
                 if __file.endswith('tiff'):
                     tiff_img_array = tifffile.imread(__file)
                     tiff_hash_img = (__file, hash(tiff_img_array), 'tiff')
                     status = self.__store_index(tiff_hash_img)
                     if status:
                        continue
                     else:
                         self.__raise_index_error()
                 else:
                    with __file.open() as img_file:
                       img = np.array(Image.open(img_file))
                       hash_img = (__file, hash(img), 'img')
                       status = self.__store_index(hash_img)
                       if status:
                          continue
                       else:
                          self.__raise_index_error()
              finally:
                 continue

           elif __file.endswith('svg'):
              rig = DataProcessor.random_int_generator()
              temp_img = f"{rig}_temp.png"
              try:
                 drawing = svg2rlg(open_file(__file))
                 renderPM.drawToFile(drawing, temp_img, fmt='PNG')
                 img = np.array(Image.open(temp_img))
                 hash_img = (__file, hash(img), 'svg')
                 status = self.__store_index(hash_img)
                 if status:
                    continue
                 else:
                    self.__raise_index_error()
              finally:
                 os.remove(temp_img)

    def __iterate_source(self):
        with self.H5File.__iter__():
            for k, v in self.H5File:
                if isinstance(v, h5tbx.Dataset):
                   yield k, hash(v)

    def __exact_match(self):
        source_file_list = list(self.__iterate_source())
        for row in source_file_list:
            k, v = row
            for k1, v1 in self.idx_db.open_db(read=True).cursor():
                if v == v1:
                    return [{True, k, v} for k, v in source_file_list if k == k1]

    def search_exact_match(self):
        match_list = self.__exact_match()
        return match_list

