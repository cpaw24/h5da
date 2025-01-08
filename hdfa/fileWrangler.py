from typing import List, Tuple, AnyStr
import gzip
from zipfile import ZipFile
from dataWrangler import H5DataCreator
import h5py as h5


class H5FileCreator:

    def __init__(self, output_file: AnyStr, write_mode: AnyStr = 'a') -> None:
        self.__output_file = output_file
        self.__write_mode = write_mode

    def create_file(self) -> h5.File:
        h5_file = h5.File(self.__output_file, mode=self.__write_mode, libver='latest', locking=True, driver='stdio')
        return h5_file


class fileHandler:

    def __init__(self, input_file: str):
        self.__input_file = input_file
        self.__data_creator = H5DataCreator

    def __file_list(self) -> List:
        if self.__input_file.endswith('zip' or 'z'):
            with ZipFile(self.__input_file, 'r') as zip:
                file_list = zip.namelist()
                return file_list
        elif self.__input_file.endswith('gz' or 'gzip'):
            with gzip.open(self.__input_file, 'rb') as gzf:
                file_list = gzip.GzipFile(fileobj=gzf).fileobj.__dict__.get('namelist')
                return file_list

    def __open_zip(self) -> Tuple[List, List]:
        zip = ZipFile(self.__input_file, 'r')
        file_list = self.__file_list()
        for file in file_list:
            content, files = self.__data_creator.classify_inputs(file, zip)
            return content, files

    def __open_h5(self) -> Tuple[List, List]:
        input_file = self.__input_file
        h5file = h5.File.file(input_file).read().decode('utf-8')
        process, local_q = self.__data_creator.classify_inputs(h5file, h5file)
        return process, local_q

    def __open_gzip(self) -> Tuple[List, List]:
        gzfile = gzip.open(self.__input_file, 'r', encoding='utf-8')
        file_list = self.__file_list()
        for file in file_list:
            process, local_q = self.__data_creator.classify_inputs(file, gzfile)
            return process, local_q

    def input_file_type(self) -> str | Tuple[List, List]:
        if self.__input_file.endswith('zip' or 'z'):
            content_list, file_list = self.__open_zip()
            return content_list, file_list
        elif self.__input_file.endswith('h5' or 'hdf5'):
            process, local_q = self.__open_h5()
            return process, local_q
        elif self.__input_file.endswith('gz' or 'gzip'):
            process, local_q = self.__open_gzip()
            return process, local_q
        else:
            return ['unknown'], ['unknown']