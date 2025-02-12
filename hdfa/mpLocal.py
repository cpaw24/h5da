from multiprocessing import Process, Manager, Queue, freeze_support
from typing import List, Type


class MpQLocal:
	"""
	Creates multiprocessing queue object, and methods to send/receive data.
	"""
	def __init__(self):
		self.__manager = Manager
		self.__queue = Queue()
		self.Process = Process
		self.Process(
			target=self.send_data,
			args=(self.__queue, ["Queue Open"])).start()
		super().__init__()

	def __setup_mp_queue(self) -> Queue:
		queue = self.__manager.Queue()
		return queue

	def get_context(self) -> Type:
		return self.get_context('spawn')

	def get_queue(self) -> Queue:
		return self.__queue

	def get_manager(self) -> Manager:
		return self.__manager

	def get_process(self) -> Process:
		return self.Process

	def join_mp_process(self, p: Process) -> None:
		return p.join()

	def send_data(self, q: Queue, data: List):
		return q.put(data)

	def recv_data(self, q: Queue) -> List:
		return q.get()

	def mp_queue_empty(self, q: Queue):
		return q.empty()

	def terminate_mp_process(self, p: Process) -> None:
		return self.Process.terminate(p)
