import os
import logging
import re
import h5py as h5
import lmdb
from click import open_file
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from data_wrangler import H5DataCreator
from typing import AnyStr, Dict, List, Tuple
import numpy as np
from PIL import Image
import tifffile


class H5SearchImages:
    def __init__(self, input_file: AnyStr, searched_files: List, use_path: AnyStr = os.getcwd()):
        self.__logger = logging.getLogger(__name__)
        self.__input_file = input_file
        self.__use_path = use_path
        self.__searched_files = searched_files
        self.__input_file_size = os.path.getsize(self.__input_file)
        self.H5File = h5.File(self.__input_file, 'r', libver='latest', driver='stdio', locking=True)
        if self.__input_file_size <= 100000000:
            self.idx_db = lmdb.open(os.path.join(self.__use_path, 'index.lmdb'), map_size=(1024 * 1024 * 102.4))
        elif self.__input_file_size > 100000000:
            self.idx_db = lmdb.open(os.path.join(self.__use_path, 'index.lmdb'), map_size=(1024 * 1024 * 1024))

    def __content_index(self):
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        __files = self.__searched_files
        for __file in __files:
            if __file.endswith(image_extensions):
                try:
                    if __file.endswith('tiff'):
                        img_array = tifffile.imread(__file)
                        hash_img = hash(img_array)
                        with self.idx_db.begin(write=True).cursor() as txn:
                            txn.put(__file, hash_img)
                            txn.put(hash_img, img_array)
                            txn.commit()
                    else:
                        with __file.open() as img_file:
                            img = Image.open(img_file)
                            img_array = np.array(img)
                            hash_img = hash(img_array)
                            with self.idx_db.begin(write=True).cursor() as txn:
                                txn.put(__file, hash_img)
                                txn.put(hash_img, img_array)
                                txn.commit()
                finally:
                    continue

            elif __file.endswith('svg'):
                rig = H5DataCreator.random_int_generator()
                temp_img = f"{rig}_temp.png"

                try:
                    drawing = svg2rlg(open_file(__file))
                    renderPM.drawToFile(drawing, temp_img, fmt='PNG')
                    img = Image.open(temp_img)
                    img_array = np.array(img)
                    hash_img = hash(img_array)
                    with self.idx_db.begin(write=True).cursor() as txn:
                        txn.put(__file, hash_img)
                        txn.put(hash_img, img_array)
                        txn.commit()

                finally:
                    os.remove(temp_img)
                    txn.commit()
                    continue

    def __iterate_source(self):
        with self.H5File.__iter__():
            for k, v in self.H5File:
                if isinstance(v, h5.Dataset):
                   yield k, hash(v)

    def __exact_match(self):
        source_file_list = list(self.__iterate_source())
        for row in source_file_list:
            k, v = row
            for k1, v1 in self.idx_db.open_db().cursor():
                if v == v1:
                    return [{True, k, v} for k, v in source_file_list if k == k1]

    @property
    def search_dataset_images(self):
        match_list = self.__exact_match()
        return match_list

