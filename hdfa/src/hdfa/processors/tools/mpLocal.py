from multiprocessing import Process, Manager, Queue, freeze_support
from typing import List, Type


class MpQLocal:
	"""
	Creates multiprocessing queue object.
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

	def send_data(self, q: Queue, data: List):
		return q.put(data)

	def terminate_mp_process(self, p: Process) -> None:
		return self.Process.terminate(p)
