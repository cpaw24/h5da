from typing import Dict, List, Tuple
import h5py as h5



class H5DataRetriever:
	def __init__(self, input_file: str, group_list: List, dataset_list: List) -> None:
		self.__input_file = input_file
		self.__group_data_list = group_list
		self.__dataset_data_list = dataset_list
		self.__h5_file = h5.File(self.__input_file, 'r', libver='latest', locking=True, driver='None')

	def recursive_retrieval(self) -> Dict | h5.Dataset | None:
		if self.__group_data_list:
			target = [g for g in self.__group_data_list if g in self.__h5_file]
		elif self.__dataset_data_list:
			target = [d for d in self.__dataset_data_list if d in self.__h5_file]

			if isinstance(target, h5.Group):
				return {k: v for k, v in target.items()}
			elif isinstance(target, h5.Dataset):
				return {k: v for k, v in target.collective.__dict__()}
			else:
				return {k: v for k, v in h5.Group(target).attrs.items()}

	def retrieve_all_data_lists(self) -> Tuple[List, List]:
		group_data_list = [self.__h5_file[name] for name in self.__h5_file if
		                   isinstance(self.__h5_file[name], h5.Group)]
		dataset_data_list = [self.__h5_file[name] for name in self.__h5_file if
		                     isinstance(self.__h5_file[name], h5.Dataset)]
		return group_data_list, dataset_data_list

	def retrieve_group_attrs_data(self) -> List[Dict]:
		attrs_list: List = []
		groups = self.retrieve_group_list()
		for g in groups:
			attrs_list.append(self.__h5_file[g].attrs.items())
		return attrs_list

	def retrieve_group_list(self) -> List:
		return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Group)]

	def retrieve_dataset_list(self) -> List:
		return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Dataset)]

	def retrieve_searched_group(self, searched_group: str) -> Tuple:
		group = self.__h5_file.get(searched_group)
		if isinstance(group, h5.Group):
			return (searched_group, {k: v for k, v in group.items()}, [i for i in group.attrs.keys() if i])

	def retrieve_searched_dataset(self, searched_dataset: str) -> Tuple:
		dataset = self.__h5_file.get(searched_dataset)
		if isinstance(dataset, h5.Dataset):
			return (searched_dataset, {item for item in dataset.chunks}, [i for i in dataset.attrs.keys() if i])