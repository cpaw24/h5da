from collections.abc import KeysView, ValuesView, ItemsView, Mapping
from hdfa.dataRetriever import H5DataRetriever
from typing import List, Dict

def retrieve_data():
    rdl = H5DataRetriever(
        input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_group_list()
    ds = H5DataRetriever(
        input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_dataset_list()
    sg = H5DataRetriever(
        input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_group_attrs_data()
    return rdl, ds, sg

def view_data(groups: List, datasets: List, group_attrs_data: List[Dict]):
    for g in groups:
        if g == 'fashion-dataset':
           group_result = H5DataRetriever(
                 input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_searched_group(g)
           dsd, rsg, attrs1 = group_result
           for kset in dsd.keys():
              print(KeysView(dsd[kset]))


    for d in datasets[0:2]:
        result = H5DataRetriever(
            input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_searched_dataset(d)

        dsf, group, attrs = result
        data = dsf[0].values[:]
        print(type(data))
        print(dsf.name, dsf.nbytes, dsf.shape, dsf.dtype, dsf.ndim, len(group))

if __name__ == '__main__':
    rdl, ds, sg = retrieve_data()
    view_data(rdl, ds, sg)
