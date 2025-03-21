import json
import h5py as h5
from typing import AnyStr, List, Tuple, Dict

class SchemaProcessor:
    def __init__(self, schema_file: AnyStr, config_file: AnyStr, output_file: AnyStr):
        self.__schema_file = schema_file
        self.__config_file = config_file
        self.__output_file = output_file

    def get_schemas(self) -> Tuple[List, List]:
        """Get input and output schemas from schema file.
        :returns tuple of input and output paths from schema file"""
        in_path_list = []
        out_path_list = []
        schema_file = self.__schema_file
        files = ['outgoing']
        schema = json.load(open(schema_file, 'r'))
        schema_file = schema.get('files')
        for file in files:
            file_path = schema_file.get(file)
            if file == 'outgoing':
                file_name = file_path.get(self.__output_file.split('/')[-1])
                if file_name:
                    output_groups = file_name.get('groups')
                    if output_groups:
                        for k, v in output_groups.items():
                            out_path_list.append([k])
                else:
                    print("No output schema to process")
        return in_path_list, out_path_list

    def write_schema(self, h5_file: h5.File, path: List,
                      content_list: List, processed_file_list: List[str]):
        try:
            flg: AnyStr = ''
            if not content_list: content_list: List = []
            content_length = content_list.count('unknown') | len(content_list)
            if not processed_file_list: processed_file_list: List = []
            for l in path:
                if isinstance(l, List):
                    for i in l:
                        flg = i
                else:
                    flg = l

                h5_file.require_group(flg)
                h5_file[flg].attrs['file_name'] = h5_file.name
                h5_file[flg].attrs['schema_file'] = self.__schema_file
                h5_file[flg].attrs['content_size'] = content_length
                print(f"write group {flg} attrs")

                content_list.append(['Group', flg])
                processed_file_list.append(flg)
            return content_list, processed_file_list

        except Exception as e:
            print(f'write_schema Exception: {e}')

    def process_schema_file(self, h5_file: h5.File, content_list: List = None,
                             processed_file_list: List = None) -> None | List | str:
        """Load and persist contents of the schema file
        :param h5_file is file object
        :param content_list is list of objects
        :param processed_file_list is list of objects
        :returns list of objects or string"""
        try:
            if not content_list: content_list: List = []
            if not processed_file_list: processed_file_list: List = []
            """ get schemas """
            h5_file.require_group('root')
            in_path, out_path = self.get_schemas()
            """ check for output schema """
            if len(out_path) > 0:
                self.write_schema(h5_file=h5_file, path=out_path,
                                   content_list=content_list, processed_file_list=processed_file_list)
                return out_path
            elif len(in_path) > 0 and len(out_path) == 0:
                self.write_schema(h5_file=h5_file, path=in_path,
                                   content_list=content_list, processed_file_list=processed_file_list)
                return in_path
            else:
                return "No schema to process"
        except Exception as e:
            print(f'process_schema_file Exception: {e, e.args}')

    def map_schema_classifications(self, classification_key: AnyStr, schema_d: Dict) -> List[str] | None:
        """ Get content type, file path, and key from schema and user classification keys. """
        return_list: List = []
        file = self.__output_file.split('/')[-1]
        groups = schema_d['files']['outgoing'][file]['groups']
        leaf_content = schema_d['files']['outgoing']['leaf_content']
        location_map = schema_d['files']['outgoing']['location_leaf_map']
        key = classification_key
        for kg, vg in groups.items():
           if key in kg:
              file_group_key = vg
              file_group = kg
        for leaf in leaf_content:
           if key == leaf:
              file_leaf_content = leaf_content[leaf]["sources"]
        for k, v in location_map.items():
           if key == k:
              file_location = v

              if file_group_key == file_location:
                 return [f"{classification_key}: {file_group}",
                          f"{classification_key}-content: {file_leaf_content}",
                          f"{classification_key}-location: {file_location}"]
