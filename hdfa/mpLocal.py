from multiprocessing import Process, Manager
from typing import List, Dict, Tuple

class mpQLocal:
	def __init__(self):
		self.Process = Process

	def __setup_mp(self) -> Manager:
		manager = Manager()
		return manager

	def __setup_mp_queue(self) -> Manager.Queue:
		manager = self.__setup_mp()
		queue = manager.Queue()
		return queue

	def setup_mp_process(self) -> Tuple[Process, Manager.Queue]:
		q = self.__setup_mp_queue()
		p = self.Process(target=q)
		pr = self.start_mp_process(p)
		return pr, q

	def start_mp_process(self, p: Process) -> Process:
		p.start()
		return p

	def join_mp_process(self, p: Process) -> Process:
		p.join()
		return p

	def terminate_mp_process(self, p: Process) -> Process:
		p.terminate()
		return p

	def send_data(self, q: Manager.Queue, data: List) -> Manager.Queue:
		q.put(data)
		return q

	def recv_data(self, q: Manager.Queue) -> List:
		data = q.get()
		return data

	def mp_queue_empty(self, q: Manager.Queue):
		return q.empty()
