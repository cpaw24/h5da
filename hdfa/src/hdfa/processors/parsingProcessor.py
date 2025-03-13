from typing import List, Dict, AnyStr, Tuple
import numpy as np


class ParsingProcessor:
    def __init__(self):
       pass

    def parse_data(self, input_dict: Dict | np.ndarray) -> List:
        """
        Recursively parses a nested dictionary or a numpy array to extract and organize
        data into a list of key-value pairs.
        :param input_dict: Dictionary or numpy array to parse.
        :return: List of: key-value pairs, numpy arrays, or tuples.
        """
        value_list: List = []
        if isinstance(input_dict, Dict):
            for k, v in input_dict.items():
                if isinstance(v, Dict):
                    """ Recursive call for nested dictionaries """
                    value_list.extend([self.parse_data(v)])
                elif isinstance(v, (List, np.ndarray)):
                    """ Check if the list or array contains integers """
                    if all(isinstance(i, int) for i in v):
                        """ Ensure v is converted to a numpy array only when needed """
                        value_list.append([k, np.ndarray(v)])
                    else:
                        """ Add raw lists if not integers """
                        value_list.append([k, v])
                elif isinstance(v, Tuple):
                    _a, _b, _c = v
                    value_list.append([k, [_a, _b, _c]])
                elif isinstance(v, (int, str)):
                    """ Add primitive types (e.g., strings, numbers) """
                    value_list.append([k, v])
        return value_list

    @staticmethod
    def find_data_size(data: List | str | int) -> int:
        """Find the maximum size of a nested list, and return the value.
        :param data: List of key-value pairs.
        :return: Maximum size of the list as integer."""
        if isinstance(data, str) | isinstance(data, int):
            return 1
        if not isinstance(data, list):
            return 0
        if not data:
            return 1
        return 1 + max(map(ParsingProcessor.find_data_size, data))

    @staticmethod
    def format_key_value(key: AnyStr, value, index=None) -> AnyStr | Tuple[AnyStr, int] | Tuple[AnyStr, int, int] | None:
        """Format key-value pairs into string.
        :param key: Key of the key-value pair.
        :param value: Value of the key-value pair.
        :param[optional] index: Index of the key-value pair.
        :return: Formatted string or tuple of strings."""
        if index is not None:
            return f"key: {key}, value: {value}, index: {index}"
        return f"key: {key}, value: {value}"

    @staticmethod
    def is_list_of_size(lst: List, size: int) -> bool | None:
        """Check if an object is a list and of a given size.
        :param lst: List object.
        :param size: Expected size of the list.
        :return: True if the list is of the given size, False otherwise."""
        return isinstance(lst, list) and len(lst) == size

    def process_row(self, row) -> List | None:
        """Process row(s) and return formatted strings. Support for nested lists.
        :param row: List of key-value pairs.
        :return: List of formatted strings or None."""
        list_size_single: int = 1  # Constant for list size one
        list_size_double: int = 2  # Constant for list of two elements
        list_size_triple: int = 3  # Constant for list size three
        result: List = []
        elements_list: List = []
        try:
            if isinstance(row, List):
               if self.is_list_of_size(row, list_size_double) | self.is_list_of_size(row, list_size_single):
                  key, value = map(str, row)
                  result.append(self.format_key_value(key, value))
               elif self.is_list_of_size(row, list_size_triple):
                  key, value, index = row
                  result.append(self.format_key_value(key, value, index))
               elif list_size_triple < len(row):
                   for element in row:
                      if (self.is_list_of_size(element, list_size_single)
                              and self.find_data_size(element) == list_size_double):
                         key, value = element
                         result.append(self.format_key_value(key, value))
                      elif self.is_list_of_size(element, list_size_double):
                         key, value = element
                         result.append(self.format_key_value(key, value))
                      elif self.is_list_of_size(element, list_size_triple):
                          key, value, index = element
                          result.append(self.format_key_value(key, value, index))
                      elif isinstance(element, List):
                          if self.find_data_size(element) > list_size_triple:
                             for inner_element in element:
                                if (self.find_data_size(inner_element) == list_size_double
                                        and not isinstance(inner_element, List)):
                                   key, value = inner_element
                                   if isinstance(value, List):
                                      for v_key, v_value in value:
                                         result.append(self.format_key_value(v_key, v_value))
                                   elif isinstance(value, Dict):
                                      for k, v in value.items():
                                         result.append(self.format_key_value(k, v))
                      elif isinstance(element, str | int | float):
                         elements_list.append(str(element))
                   result.append(self.format_key_value('List-of-Values', elements_list))
            return result

        except Exception as e:
           print(f'ParsingProcessor Exception: {e, e.args}')
        except ValueError as ve:
            print(f'ParsingProcessor ValueError: {ve, ve.args}')
        except TypeError as te:
            print(f'ParsingProcessor TypeError: {te, te.args}')

    def find_list_depth(self, obj: List | AnyStr) -> int | None:
        """Find the maximum depth of a nested list, and return the value.
        :param obj: List or string.
        :return: Maximum depth of the list as integer."""
        if not isinstance(obj, list):
            return 0
        if not obj:
            return 1
        return 1 + max(self.find_list_depth(item) for item in obj)