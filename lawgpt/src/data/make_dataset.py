import os
import json


class MakeDataset_EurLexSumIt:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _get_data(self, file_name: str):
        test_path = os.path.join(self.root_dir, file_name)
        if not os.path.exists(test_path):
            raise FileNotFoundError

        data_list = []
        with open(test_path, "r") as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    data_list.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        return data_list

    def test_data(self, test_file_name: str):
        return self._get_data(test_file_name)

    def train_data(self, train_file_name: str):
        return self._get_data(train_file_name)

    def dev_data(self, dev_file_name: str):
        return self._get_data(dev_file_name)
