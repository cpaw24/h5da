import logging
import re

import h5py as h5
from functools import lru_cache
from typing import AnyStr, Dict, List, Tuple



class H5DataRetriever:
    def __init__(self, input_file: str, group_list: List, dataset_list: List) -> None:
        self.__input_file = input_file
        self.__group_data_list = group_list
        self.__dataset_data_list = dataset_list
        self.__logger = logging.getLogger(__name__)
        self.__h5_file = h5.File(self.__input_file, 'r', libver='latest', locking=True, driver='None')

    def recursive_retrieval(self) -> List | None:
        target: List = []
        return_list: List = []
        if isinstance([group for group in self.retrieve_group_list()], h5.Group):
            target.append([g for g in self.retrieve_group_list() if g in self.__h5_file[g]])
        elif isinstance([ds for ds in self.retrieve_dataset_list()], h5.Dataset):
            target.append([d for d in self.retrieve_dataset_list() if d in self.__h5_file[d]])

        if isinstance([t for t in target], h5.Group):
            group_dict = {k: v for k, v in [t for t in target if t.items()]}
            return_list.append({"groups": group_dict})
        elif isinstance([t for t in target], h5.Dataset):
            ds_dict = {k: v for k, v in [t for t in target if t.collective.__dict__()]}
            return_list.append({"datasets": ds_dict})

        return return_list

    def retrieve_all_data_lists(self) -> Tuple[List, List]:
        group_data_list = self.retrieve_group_list()
        dataset_data_list = self.retrieve_dataset_list()
        return (group_data_list, dataset_data_list)

    def retrieve_group_attrs_data(self) -> List[Dict]:
        attrs_list: List = []
        groups = self.retrieve_group_list()
        for g in groups:
            attrs_list.append(self.__h5_file[g].attrs.items())
        return attrs_list

    @lru_cache(maxsize=256)
    def retrieve_group_list(self) -> List:
        return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Group)]

    @lru_cache(maxsize=256)
    def retrieve_dataset_list(self) -> List:
        return [name for name in self.__h5_file if isinstance(self.__h5_file[name], h5.Dataset)]

    @staticmethod
    def __search_str_validation(search_str: AnyStr) -> bool:
        if re.search(['A-Z', 'a-z', '-', '_'][0-9], search_str):
            return True
        else:
            return False

    @staticmethod
    def __search_group_priority(first_size: int, second_size: int) -> bool:
        if (first_size == second_size) or (first_size > second_size != 0):
            return True
        elif first_size < second_size != 0:
            return False

    def retrieve_searched_group(self, searched_group: AnyStr) -> Tuple:
        all_groups_size = len(self.retrieve_group_list())
        target_group_size = len(self.__group_data_list)
        group = None
        if self.__search_str_validation(searched_group):
            result = self.__search_group_priority(all_groups_size, target_group_size)
            if result:
                group = self.__group_data_list
            elif not result:
                group = self.retrieve_group_list()

            for g in group:
                if isinstance(g, h5.Group) and g.name == searched_group:
                    return (searched_group, {k: v for k, v in g.items()}, {k: v for k, v in g.attrs.items()})
        else:
            self.__logger.error(f"Invalid search string: {searched_group}")

    def retrieve_searched_dataset(self, searched_dataset: AnyStr) -> Tuple:
        all_datasets_size = len(self.retrieve_dataset_list())
        target_dataset_size = len(self.__dataset_data_list)
        dataset = None
        if self.__search_str_validation(searched_dataset):
            result = self.__search_group_priority(all_datasets_size, target_dataset_size)
            if result:
                dataset = self.__dataset_data_list
            elif not result:
                dataset = self.retrieve_dataset_list()

            for ds in dataset:
                if isinstance(ds, h5.Dataset) and ds.name == searched_dataset:
                    if ds.chunks:
                        return (searched_dataset, [item for item in ds.iter_chunks() if item], {k: v for k, v in ds.attrs.items()})
                    elif not ds.chunks:
                        return (searched_dataset, {k: v for k, v in ds.collective.__dict__()}, {k: v for k, v in ds.attrs.items()})
        else:
            self.__logger.error(f"Invalid dataset name: {searched_dataset}")
