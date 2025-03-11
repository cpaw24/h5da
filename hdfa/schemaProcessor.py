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
                            out_path_list.append([v])
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

    def map_schema_classifications(self, classification_keys: List, schema_d: Dict, config_d: Dict) -> Tuple:
        """ Get content type, file path, and key from schema and user classification keys. """
        image_types = config_d['image_extensions']
        video_types = config_d['video_extensions']
        groups = schema_d['files']['outgoing'][self.__output_file.split('/')[-1]]['groups']
        leaf_content = schema_d['files']['outgoing'][self.__output_file.split('/')[-1]]['leaf_content']

        for group, content in zip(groups, leaf_content):
           """ Check if content key is in classification keys and if it's in the group key."""
           """ Performs a fuzzy or exact match depending upon the structure of the schema."""
           if (content.key() in group.key()) and (content.key() in classification_keys):
              file_path =  group.value()
              source = leaf_content[content.key()]['sources']
              if source in image_types:
                 content_type = 'image'
              elif source in video_types:
                 content_type = 'video'
           elif group.key() in classification_keys:
               file_path =  group.value()
               source = leaf_content[group.key()]['sources']
               if source in image_types:
                  content_type = 'image'
               elif source in video_types:
                  content_type = 'video'
               else:
                  content_type = group.key()

                  return content.key, file_path, content_type
           else:
              return None, None, None
