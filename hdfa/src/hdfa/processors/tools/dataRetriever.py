import logging
import re
import h5rdmtoolbox as h5tbx
from pathlib import Path
import h5py as h5
from functools import lru_cache
from typing import Any, AnyStr, Dict, List, Tuple
from h5py import Dataset, Datatype, ExternalLink, Group, HardLink, SoftLink


class DataRetriever:
    """Inits DataRetriever class. Uses the h5rdmtoolbox library to retrieve data from the input file."""
    def __init__(self, input_file: str, group_list: List = None, dataset_list: List = None) -> None:
        self.__input_file = input_file
        self.__group_data_list = group_list
        self.__dataset_data_list = dataset_list
        self.__file_path = Path(self.__input_file)
        self.__initialize()

    def __initialize(self):
        self.__h5_file_tbx = h5tbx.File(self.__file_path, 'r')
        self.__logger = logging.getLogger(__name__)

    def recursive_retrieval(self) -> List | None:
        target: List = []
        return_list: List = []
        if isinstance([group for group in self.retrieve_group_list()], h5.Group):
            target.append([g for g in self.retrieve_group_list() if g in self.__h5_file_tbx[g]])
        elif isinstance([ds for ds in self.retrieve_dataset_list()], h5.Dataset):
            target.append([d for d in self.retrieve_dataset_list() if d in self.__h5_file_tbx[d]])

        if isinstance([t for t in target], h5.Group):
            group_dict = {k: v for k, v in [t for t in target if t.items()]}
            return_list.append({"groups": group_dict})
        elif isinstance([t for t in target], h5.Dataset):
            ds_dict = {k: v for k, v in [t for t in target if t.collective.__dict__()]}
            return_list.append({"datasets": ds_dict})

        return return_list

    def retrieve_all_data_lists(self) -> Tuple[List, List]:
        """Retrieves all group and dataset names from the input file."""
        group_data_list = self.retrieve_group_list()
        dataset_data_list = self.retrieve_dataset_list()
        return (group_data_list, dataset_data_list)

    def retrieve_all_group_attrs_data(self) -> List[Dict]:
        """Retrieves all group attributes from the input file."""
        attrs_list: List = []
        groups = self.retrieve_group_list()
        for g in groups:
            attrs_list.append([g, self.__h5_file_tbx[g].attrs.items()])
        return attrs_list

    @lru_cache(maxsize=25)
    def retrieve_group_list(self) -> List:
        """Retrieves all group names from the input file."""
        return [name for name in self.__h5_file_tbx if isinstance(self.__h5_file_tbx[name], h5.Group)]

    @lru_cache(maxsize=25)
    def retrieve_dataset_list(self) -> List | None:
        """Retrieves all dataset names from the input file."""
        group_list = self.retrieve_group_list()
        for g in group_list:
           group = self.__h5_file_tbx[g]
           return [name for name in group if isinstance(group[name], h5.Dataset)]

    @staticmethod
    def __search_str_validation(search_str: AnyStr) -> bool:
        if re.search('\\w+', search_str, re.UNICODE):
            return True
        else:
            return False

    @staticmethod
    def __search_group_priority(first_size: int, second_size: int) -> bool | None:
        """Not Implemented."""
        if (first_size == second_size) or (first_size > second_size != 0):
            return True
        elif first_size < second_size != 0:
            return False

    def retrieve_searched_group(self, searched_group: AnyStr = None) -> Tuple[Group, Any, List[Any]] | None:
        """Retrieves the searched group from the input file."""
        if self.__search_str_validation(searched_group):
            result = None
            if result:
                group_list = self.__group_data_list
            elif not result:
                group_list = self.retrieve_group_list()
                for g in group_list:
                    group = self.__h5_file_tbx[g]
                    if isinstance(group, h5.Group) and group.name.replace('/', '') == searched_group:
                        return (group, group.get(searched_group, getclass=True, getlink=True),
                                [group.attrs.items() for group in group.values()])
        else:
            self.__logger.error(f"Invalid search string: {searched_group}")

    def retrieve_searched_dataset(self, searched_dataset: AnyStr) -> List | None:
        """Retrieves the searched dataset from the input file."""
        group_list = self.retrieve_group_list()
        if self.__search_str_validation(searched_dataset):
           dataset = self.retrieve_dataset_list()
           for ds in dataset:
              if ds == searched_dataset:
                 for group in group_list:
                    htb = self.__h5_file_tbx[group][ds]
                    return [htb, self.__h5_file_tbx[group], self.__h5_file_tbx[group].attrs.items()]

        else:
            self.__logger.error(f"Invalid dataset name: {searched_dataset}")

