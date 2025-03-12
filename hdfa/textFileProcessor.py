from typing import AnyStr, List, Tuple
import gzip
import h5
import tarfile
import zipfile
import json
import csv
import os
from zipfile import ZipFile

class TextFileProcessor:
    def __init__(self):
        pass

    @staticmethod
    def process_json(file: AnyStr, open_file: ZipFile | gzip.GzipFile,
                     content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]] | None:
        """Converts JSON files to dictionaries
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return tuple[content_list, processed_file_list]"""
        try:
            file_name = file.casefold()
            if isinstance(open_file, tarfile.TarFile):
                raw_content = open_file.extractfile(file).read().decode('utf-8').splitlines()
            else:
                raw_content = open_file.read(file).decode('utf-8').splitlines()

            content = [row for row in raw_content]
            content = json.loads(content[0])
            line_count = len(content_list)
            content_list.append([file_name, content, line_count])
            processed_file_list.append(file_name + '-' + str(line_count))

            if content_list and processed_file_list:
                return content_list, processed_file_list

        except Exception as e:
            print(f'process_json Exception: {e}')
        finally:
            """ Ensure temp file is removed. """
            if os.path.exists('/' + file_name):
                os.remove('/' + file_name)

    @staticmethod
    def process_csv(file: AnyStr, open_file: ZipFile | gzip.GzipFile | tarfile.TarFile,
                    content_list: List, processed_file_list: List[str]) -> Tuple[List, List[str]] | None:
        """Converts CSV files to lists of values
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return tuple[content_list, processed_file_list]"""
        try:
            if isinstance(open_file, tarfile.TarFile):
                csv_file = open_file.extractfile(member=file).read().decode('utf-8').splitlines()
                csv_reader = csv.reader(csv_file, delimiter=",", quotechar='"', doublequote=True)
            elif isinstance(open_file, zipfile.ZipFile):
                csv_file = open_file.open(file, force_zip64=True)
                csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",",
                                        doublequote=True, quotechar='"')
            else:
                csv_file = open_file.open(file)
                csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines(), delimiter=",", )

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
        finally:
            if os.path.exists('/' + file):
                os.remove('/' + file)