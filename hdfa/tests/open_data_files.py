from collections.abc import KeysView, ValuesView, ItemsView, Mapping
import h5py

from hdfa.dataRetriever import H5DataRetriever
from typing import List, Dict

def retrieve_data():
    rdl = H5DataRetriever(
        input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//fashion-dataset_filestore.h5').retrieve_group_list()
    ds = H5DataRetriever(
        input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//fashion-dataset_filestore.h5').retrieve_dataset_list()
    sg = H5DataRetriever(
        input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//fashion-dataset_filestore.h5').retrieve_group_attrs_data()
    return rdl, ds, sg
#
#
def view_data(groups: List, datasets: List, group_attrs_data: List[Dict]):
    for g in groups:
        if g == 'fashion-dataset':
           group_result = H5DataRetriever(
                 input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//fashion-dataset_filestore.h5').retrieve_searched_group(g)
           dsd, attrs1, rsg = group_result
           for kset in dsd.keys():
              KeysView(dsd[kset])


    for d in datasets[0:1]:
        result = H5DataRetriever(
            input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//fashion-dataset_filestore.h5').retrieve_searched_dataset(d)

        dsf, group = result
        data = dsf[0].values[:]
        print(type(data))
        print(dsf.name, dsf.nbytes, dsf.shape)
        for key, items in zip(KeysView(group.attrs.keys()), ItemsView(group.attrs)):
            print(key, items)


if __name__ == '__main__':
    rdl, ds, sg = retrieve_data()
    view_data(rdl, ds, sg)
