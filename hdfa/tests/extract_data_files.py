from collections.abc import KeysView, ValuesView, ItemsView
from hdfa.dataWrangler import H5DataCreator
import h5py as h5
from hdfa.dataRetriever import H5DataRetriever

#["//Volumes//ExtShield//datasets//archive.zip"] #,
input_files = ["//Volumes//ExtShield//datasets//fashion-dataset.zip"]

for input_file in input_files:
  file_name = input_file.split('/')[-1].split('.')[0]
  H5DataCreator(input_file=input_file, input_dict={},
                output_file=f'tests//{file_name}_filestore.h5', schema_file='schema.json').input_processor()

# rdl = H5DataRetriever(input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//archive_filestore.h5',
#                       group_list=[], dataset_list=[]).retrieve_group_list()
# sg = H5DataRetriever(input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//archive_filestore.h5',
#                       group_list=[], dataset_list=[]).retrieve_group_attrs_data()
# for g in rdl:
#   f, rsg, attrs = H5DataRetriever(input_file='//Users//claude_paugh//PycharmProjects//HDF5//hdfa//tests//archive_filestore.h5',
#                         group_list=[], dataset_list=[]).retrieve_searched_group(g)
# rdl = H5DataRetriever(input_file='./archive_2_filestore.h5', group_list=[], dataset_list=[]).retrieve_group_list()
# print(rdl)
# for view in sg:
#   for r in enumerate(view):
#     i, d = r
#     ix, val = d
#     print(i)
#     print(val)



