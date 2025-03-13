from collections.abc import KeysView
from src.hdfa.processors.tools.dataRetriever import DataRetriever
import time
from typing import List, Dict

def retrieve_metadata():
    rdl = DataRetriever(
        input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_group_list()
    ds = DataRetriever(
        input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_dataset_list()
    sg = DataRetriever(
        input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_group_attrs_data()
    return rdl, ds, sg

def view_data(groups: List, datasets: List, group_attrs_data: List[Dict]):
    for g in groups:
        if g == 'fashion-dataset':
           group_result = DataRetriever(
                 input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_searched_group(g)
           dsd, rsg, attrs1 = group_result
           for kset in dsd.keys():
              print(KeysView(dsd[kset]))

    for d in datasets[10:30]:
        result = DataRetriever(
            input_file='//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5').retrieve_searched_dataset(d)

        dsf, group, attrs = result
        data = dsf[0].values[:]
        print(type(data))
        print(dsf.name, dsf.nbytes, dsf.shape, dsf.dtype, dsf.ndim, len(group))

if __name__ == '__main__':
    rdl, ds, sg = retrieve_metadata()
    print(len(rdl), len(ds))
    time.sleep(3)
    print(
        "The groups in the file are: ",
        rdl,
        "\nThe datasets in the file are: ",
        ds
    )
    for g in rdl:
        print(g)
        time.sleep(1)
    for d in ds:
        print(d.name, d.nbytes, d.shape, d.dtype, d.ndim, len(d))
        time.sleep(1)
    view_data(rdl, ds, sg)
