import gzip
import io
import tarfile
import zipfile
from typing import AnyStr, List
import numpy as np
import h5py as h5
from skvideo import io as vio


class VideoProcessor:
	def __init__(self):
		pass

	@staticmethod
	def process_video(file: AnyStr | io.BytesIO,
	                  open_file: zipfile.ZipFile | h5.File | gzip.GzipFile | tarfile.TarFile | io.BytesIO,
	                  content_list: List, processed_file_list: List[str], pix_format: AnyStr = "rgb24") -> List[List] | None:
		"""Converts MP4/MP3 video to Numpy array
		:param pix_format:
		:param file: Path to the input file (e.g., .zip, .gz, .h5).
		:param open_file: ZipFile, Gzip, or h5.File object.
		:param content_list: Content list to be processed.
		:param processed_file_list: Processed file list to be processed.
		:return list[content_list, processed_file_list]"""
		try:
			import cv2
			frames = []
			if isinstance(file, str):
				ds = file.split('/')[0]
				video_data = vio.vread(file)
				shape = video_data.shape
				edges = np.zeros(shape[0:3])
				for i in range(shape[0]):
					edges[i] = cv2.Canny(video_data[i].edges, 100, 200)
				edge_output = vio.vwrite(file, edges)
				v = vio.vread(file, num_frames=shape[0], outputdict={"-pix_fmt": f"{pix_format}"})
				output_data = np.array(v.vwrite()).astype(np.uint8)
				content_list.append([edge_output, output_data, file])
				processed_file_list.append(file)
				return [content_list, processed_file_list]

			elif isinstance(file, io.BytesIO) and isinstance(open_file, io.BytesIO):
				with open_file.getbuffer() as buff_vid:
					vid_arr = np.array(np.frombuffer(buff_vid, dtype=np.uint8))
					with cv2.VideoCapture.retrieve(image=vid_arr, flag=0) as cap:
						while cap.isOpened():
							ret, frame = cap.read()
							if not ret:
								break
							frames.append(frame)

							cap.release()
							final_frame = np.dstack(frames)
							content_list.append([file.name, final_frame, file])
							processed_file_list.append(file.name)
							return [content_list, processed_file_list]

		except Exception as e:
			print(f'process_video Exception: {e}')

	def create_video_summary(self):
		pass