import logging
import re
from src.hdfa.processors.parsingProcessor import ParsingProcessor
import h5rdmtoolbox as h5tbx
from pathlib import Path
import h5py as h5
from typing import Any, AnyStr, Dict, List, Tuple


class DataRetriever:
    """Inits DataRetriever class. Uses the h5rdmtoolbox library to retrieve data from the input file."""

    def __init__(self, input_file: str) -> None:
        self.__input_file = input_file
        self.__file_path = Path(self.__input_file)
        self.__initialize()

    def __initialize(self):
        self.__h5_file_tbx = h5tbx.File(self.__file_path, 'r')
        self.__logger = logging.getLogger(__name__)
        self._parse_pro = ParsingProcessor()

    def recursive_retrieval(self) -> List | None:
        target: List = []
        return_list: List = []
        if isinstance([group for group in self.retrieve_groups_list()], h5.Group):
            target.append([g for g in self.retrieve_groups_list() if g in self.__h5_file_tbx[g]])
        elif isinstance([ds for ds in self.retrieve_all_datasets_list()], h5.Dataset):
            target.append([d for d in self.retrieve_all_datasets_list() if d in self.__h5_file_tbx[d]])

        if isinstance([t for t in target], h5.Group):
            group_dict = {k: v for k, v in [t for t in target if t.items()]}
            return_list.append({"groups": group_dict})
        elif isinstance([t for t in target], h5.Dataset):
            ds_dict = {k: v for k, v in [t for t in target if t.collective.__dict__()]}
            return_list.append({"datasets": ds_dict})

        return return_list

    def retrieve_all_data_lists(self) -> Tuple[List, List]:
        """Retrieves all group and dataset names from the input file."""
        group_data_list = self.retrieve_groups_list()
        dataset_data_list = self.retrieve_all_datasets_list()
        return (group_data_list, dataset_data_list)

    def retrieve_all_group_attrs_data(self, slice_range: Tuple = None) -> List[Dict]:
        """Retrieves all group attributes from the input file."""
        attrs_list: List = []
        if slice_range:
            start, end = slice_range
            groups = self.retrieve_groups_list()[start:end]
        else:
            groups = self.retrieve_groups_list()

        for g in groups:
            attrs_list.append([g, self.__h5_file_tbx[g].attrs.items()])
        return attrs_list

    def retrieve_groups_list(self) -> List | None:
        group_list = [name for name in self.__h5_file_tbx if isinstance(self.__h5_file_tbx[name], h5tbx.Group)]
        return group_list

    def retrieve_recursive_group_list(self, group_list: List = None) -> List | None:
        """Retrieves all group names from the input file."""
        if not group_list:
            group_list = self.retrieve_groups_list()
        else:
            group_list = group_list

        for g in group_list:
            depth = self._parse_pro.find_list_depth(g)
            print(depth)
            if depth == 1:
                for g1 in g:
                    if isinstance(g1, h5tbx.Group):
                        group_list.extend(self.__h5_file_tbx[g1])
            elif depth > 1:
                depth_count = depth - 1
                while depth < depth_count:
                    for i in range(depth - 1):
                        if isinstance(self.__h5_file_tbx[g][i], h5tbx.Group):
                            group_list.extend(self.__h5_file_tbx[g][i])
                        i += 1
                        depth_count -= 1
            elif depth == 0:
                group_list.append(g)
            else:
                self.__logger.error(f"Invalid group name: {g}")
                return None
        return group_list

    def retrieve_all_datasets_list(self, slice_range: Tuple = None) -> List | None:
        """Retrieves all dataset names from the input file."""
        if not slice_range:
            group_list = self.retrieve_groups_list()
        else:
            start, end = slice_range
            group_list = self.retrieve_groups_list()[start:end]

        ds_list: List = []
        ds_obj_list: List = []
        for g in group_list:
            depth = self._parse_pro.find_list_depth(g)
            if depth == 0:
                group = self.__h5_file_tbx[g]
                ds = [ds for ds in group if isinstance(group[ds], h5tbx.Dataset) and ds not in ds_list]
                if ds:
                    for dsd in ds:
                        ds_obj = group[dsd]
                        ds_obj_list.append([group, ds_obj, group.get(ds_obj.name, getclass=True, getlink=True)])
            else:
                group = self.__h5_file_tbx[g]
                for i in range(depth - 1):
                    ds = [ds for ds in group if isinstance(group[ds], h5tbx.Dataset) and ds not in ds_list]
                    if ds:
                        for dsd in ds:
                            ds_obj = group[dsd]
                            ds_obj_list.append([group, ds_obj, group.get(ds_obj.name, getclass=True, getlink=True)])
                    i += 1
        return ds_obj_list

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

    def retrieve_searched_group(self, searched_group: AnyStr = None, user_group_list: List = None) \
            -> List[List[Any] | Any] | None:
        """Retrieves the searched group from the input file."""
        if self.__search_str_validation(searched_group):
            if user_group_list:
                group_list = user_group_list
            else:
                result_group = [sg for sg in self.retrieve_groups_list()
                                if isinstance(self.__h5_file_tbx[sg], h5tbx.Group)]
                if len(result_group) == 1:
                    group = self.__h5_file_tbx[searched_group]
                else:
                    group = result_group

                    if searched_group in group:
                            grp_ds: List = []
                            grp_obj = self.__h5_file_tbx[searched_group]
                            grp_ds_list = [name for name in self.__h5_file_tbx[searched_group]
                                      if isinstance(self.__h5_file_tbx[searched_group][name], h5tbx.Dataset)]
                            for ds in grp_ds_list:
                                ds = self.__h5_file_tbx[searched_group][ds]
                                grp_ds.append(ds)
                            return [grp_ds, grp_obj, grp_obj.attrs.items()]
        else:
            self.__logger.error(f"Invalid search string: {searched_group}")

    def retrieve_searched_dataset(self, searched_dataset: AnyStr, target_group: h5tbx.Group | List = None) -> List | None:
        """Retrieves the searched dataset from the input file."""
        if target_group:
            group_list = target_group
        else:
            group_list = self.retrieve_groups_list()

        if self.__search_str_validation(searched_dataset):
            for group in group_list:
               if isinstance(self.__h5_file_tbx[group], h5tbx.Group):
                  ds_list = [name for name in self.__h5_file_tbx[group]
                             if isinstance(self.__h5_file_tbx[group][name], h5tbx.Dataset)]
                  if searched_dataset in ds_list:
                     htb = self.__h5_file_tbx[group][searched_dataset]

                     if htb:
                        return [self.__h5_file_tbx[group][searched_dataset],
                                self.__h5_file_tbx[group],
                                self.__h5_file_tbx[group][searched_dataset].attrs.items()]

        else:
            self.__logger.error(f"Invalid dataset name: {searched_dataset}")
