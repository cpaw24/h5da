import h5py as h5
from zipfile import ZipFile
from typing import AnyStr, Dict, List
import numpy as np
from PIL import Image
import json
import random
from data_extract import DataExtractor

input_files = ["//Volumes//ExtShield//datasets//archive.zip",
			  "//Volumes//ExtShield//datasets//archive_2.zip"]

def random_int() -> int:
	return random.randint(0, 1000000000)

for input_file in input_files:
	file_name = input_file.split('/')[-1].split('.')[0]
	de = DataExtractor(output_file=f"data_{file_name}.h5", input_file=input_file, input_dict={})



