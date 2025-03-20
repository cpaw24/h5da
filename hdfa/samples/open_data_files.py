from collections.abc import KeysView, ItemsView, ValuesView, Iterable
import numpy as np
from src.hdfa.processors.tools.dataRetriever import DataRetriever
import time
from typing import List, Dict

def retrieve_metadata():
    data_retriever = DataRetriever('//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5')
    rdl = data_retriever.retrieve_group_list()[50:60]
    ds = data_retriever.retrieve_dataset_list(slice_range=(50, 60))
    sg = data_retriever.retrieve_all_group_attrs_data(slice_range=(50, 60))
    return rdl, ds, sg

def view_data(groups: List, datasets: List, group_attrs_data: List[Dict]):
    data_retriever = DataRetriever('//Volumes//ExtShield//datasets//hdfa_files//fashion-dataset_filestore.h5')
    for g in groups:
       group_result = data_retriever.retrieve_searched_group(searched_group=g)
       dsd, rsg, attrs1 = group_result

       for d in dsd:
          ds_name = d.name.split("/")[-1]
          result = data_retriever.retrieve_searched_dataset(searched_dataset=ds_name)
          dsf, group, attrs = result
          data = dsf.values[:]
          print(type(data))
          data_bytes = np.array(data).astype(dtype='S')
          print(data_bytes)
          print(dsf.name, dsf.nbytes, dsf.shape, dsf.dtype, dsf.ndim, len(group))

       for a in attrs1:
          print(KeysView(a))
          print(ItemsView(a))
          print(ValuesView(a))
          print(a)
          print(type(a))
          print(a.keys())
          print(a.values())
          print(a.items())
          print(a.get('name'))
          print(a.get('shape'))

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
    for g in rdl[0:10]:
        print(g)
    for d in ds[0:10]:
        group, ds, at = d
        print(type(ds))
        hdf5_ds = ds.values[:]
        print(hdf5_ds)
    view_data(rdl, ds, sg)
