import zipfile
import h5
import gzip
import tifffile
import tarfile
import os
import io
import random
from PIL import Image
from typing import AnyStr, List, Dict
import numpy as np
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg


class ImageProcessor:
    def __init__(self,  output_file: AnyStr, schema_dict: Dict = None, config_dict: Dict = None):
        self.__output_file = output_file
        self.__schema_dict = schema_dict
        self.__config_dict = config_dict

    def convert_images(self, file: AnyStr | io.BytesIO,
                       open_file: zipfile.ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile | io.BytesIO,
                       content_list: List, processed_file_list: List[str]):
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
              except Exception as e:
                 print(e)

           if file.endswith('tiff'):
              with open_file.open(file) as tiff_file:
                 img = tifffile.imread(tiff_file)
                 content_list.append([ds, img, tiff_file])
                 processed_file_list.append(file)
           else:
              with open_file.open(file) as img_file:
                 img = np.array(Image.open(img_file))
                 content_list.append([ds, img, file])
                 processed_file_list.append(file)

              return content_list, processed_file_list

        except Exception as e:
           print(f'convert_images Exception: {e}')

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

    def process_bio_image(self):
        pass

    @staticmethod
    def random_int_generator() -> str:
       random_int = random.randint(1, 1000000)
       return str(random_int)

