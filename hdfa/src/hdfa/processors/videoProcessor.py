import gzip
import io
import tarfile
import zipfile
from typing import AnyStr, List, Tuple
from PIL import Image
import av, av.datasets
import numpy as np

class VideoProcessor:
    def __init__(self):
        pass

    @staticmethod
    def process_video(file: AnyStr | io.BytesIO,
                      open_file: zipfile.ZipFile | gzip.GzipFile | tarfile.TarFile | io.BytesIO,
                      content_list: List, processed_file_list: List) -> Tuple[List, List] | None:
        """Converts MP4/MP3 video to Numpy array
        :param pix_format:
        :param file: Path to the input file (e.g., .zip, .gz, .h5).
        :param open_file: ZipFile, Gzip, or h5.File object.
        :param content_list: Content list to be processed.
        :param processed_file_list: Processed file list to be processed.
        :return list[content_list, processed_file_list]"""
        try:
            if isinstance(file, str):
                ds = file.split('/')[0]
                container = av.open(open_file.open(ds), buffer_size=1024 * 1024 * 10)
                container.streams.video[0].thread_type = 'AUTO'
                columns = []
                for frame in container.decode(video=0):
                    Array = frame.to_ndarray(format='rgb24')
                    column = Array.mean(axis=1)
                    column = column.clip(0, 255).astype('uint8')
                    column = column.reshape(-1, 2, 3)
                    columns.append(column)
                container.close()

                all = np.hstack(columns)
                full = Image.fromarray(all, 'RGB')
                full = full.resize((1024, 768))
                full = np.array(full)

                content_list.append([ds, full, file])
                processed_file_list.append(file)

                return content_list, processed_file_list

        except Exception as e:
            print(f'process_video Exception: {e}')

    def create_video_summary(self):
        pass