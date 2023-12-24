from traceback import print_tb
from MakeDataset_Protocol import MakeDataset
from typing import List, Dict
import os
import json
from make_dataset_utils import copy_json_files, create_folder_if_not_exists

import pyarrow.parquet as pq
import pandas as pd
import xml.etree.ElementTree as ET

ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1"]


class MakeDataset_EurLexSumIt(MakeDataset):
    def __init__(
        self,
        root_dir: str,
        train_file_name: str = "train.json",
        test_file_name: str = "test.json",
        valid_file_name: str = "validation.json",
    ):
        self.root_dir = root_dir
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.valid_file_name = valid_file_name
        self._name = "EurLexSumIt"

    @property
    def name(self) -> str:
        return self._name

    def create_data(self, destination_path: str) -> None:
        create_folder_if_not_exists(destination_path)
        copy_json_files(self.root_dir, destination_path)
        print(f"Dataset created in {destination_path}.")


class MakeDataset_EuroParl(MakeDataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._name = "EuroParl"

    @property
    def name(self) -> str:
        return self._name

    def create_data(self, destination_path: str) -> None:
        create_folder_if_not_exists(destination_path)
        list_dataset = self._get_data_ita()
        with open(os.path.join(destination_path, "europarl-it.txt"), "w") as file:
            # Write each string from the list to the file
            for string in list_dataset:
                file.write(string + "\n")

    def _get_data_ita(self, file_name: str = "europarl-v7.it-en.it") -> List[str]:
        union_path = os.path.join(self.root_dir, file_name)
        if not os.path.exists(union_path):
            raise FileNotFoundError
        with open(union_path, "r") as file:
            lines_list = [line.strip() for line in file]
        return lines_list


class MakeDataset_Itacasehold(MakeDataset):
    def __init__(
        self,
        root_dir: str,
        train_file_name: str = "train-itacasehold.parquet",
        test_file_name: str = "test-itacasehold.parquet",
        valid_file_name: str = "validation-itacasehold.parquet",
    ):
        self.root_dir = root_dir
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.validation_file_name = valid_file_name
        self._name = "Itacasehold"

    @property
    def name(self) -> str:
        return self._name

    def _get_data(self, data_path: str) -> List[Dict[str, str]]:
        if not os.path.exists(data_path):
            raise FileNotFoundError

        table = pq.read_table(data_path)
        data_list_dict = table.to_pandas().to_dict(orient="records")
        return data_list_dict

    def _write_json_file(
        self,
        destination_path: str,
        file_name: str,
        data_list_dict: List[Dict[str, str]],
    ) -> None:
        with open(os.path.join(destination_path, file_name), "w") as json_file:
            json.dump(data_list_dict, json_file, indent=4)
        print(f"File {file_name} created in {destination_path}.")

    def create_data(self, destination_path: str) -> None:
        train_path = os.path.join(self.root_dir, self.train_file_name)
        test_path = os.path.join(self.root_dir, self.test_file_name)
        validation_path = os.path.join(self.root_dir, self.validation_file_name)

        create_folder_if_not_exists(destination_path)
        self._write_json_file(
            destination_path, "train_Itacasehold.json", self._get_data(train_path)
        )
        self._write_json_file(
            destination_path, "test_Itacasehold.json", self._get_data(test_path)
        )
        self._write_json_file(
            destination_path,
            "validation_Itacasehold.json",
            self._get_data(validation_path),
        )


class MakeDataset_Costituzionale(MakeDataset):
    def __init__(
        self,
        root_path: str,
        massime_folder_name: str = "massime",
        pronuncie_folder_name: str = "pronunce",
    ):
        self.root_path = root_path
        self.path_massime = os.path.join(root_path, massime_folder_name)
        self.path_pronuncie = os.path.join(root_path, pronuncie_folder_name)
        self._name = "Costituzionale"

    @property
    def name(self) -> str:
        return self._name

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)

        data_list_dict_massime = self._combine_json_files(self.path_massime)
        data_list_dict_pronuncie = self._combine_json_files(self.path_pronuncie)

        print("*" * 50)
        print(type(data_list_dict_massime))
        print(len(data_list_dict_massime))
        print(type(data_list_dict_massime[0]))
        print(data_list_dict_massime[1])
        print("*" * 50)

        self._write_json_file(destination_path, "massime.json", data_list_dict_massime)
        self._write_json_file(
            destination_path, "pronuncie.json", data_list_dict_pronuncie
        )

    def _write_json_file(
        self,
        destination_path: str,
        file_name: str,
        data_list_dict: List[Dict[str, str]],
    ) -> None:
        with open(os.path.join(destination_path, file_name), "w") as json_file:
            json.dump(data_list_dict, json_file, indent=4)
        print(f"File {file_name} created in {destination_path}.")

    def _combine_json_files(self, folder_path: str):
        # Initialize an empty list to store data from JSON files
        all_data = []

        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)

                # Open each JSON file and append its content to the all_data list
                for enc in ENCODINGS:
                    try:
                        with open(file_path, "r", encoding=enc) as file:
                            data = json.load(file)
                            all_data.append(data)
                            print(
                                f"Correctly read file: {file_path} with encoding: {enc}"
                            )
                    except:
                        print(f"Error reading file: {file_path} with encoding: {enc}")

        return all_data

    def union_massime_and_pronunce(
        self, massime_path: str, pronunce_path: str, destination_path: str
    ):
        all_data = []
        for file_path in [massime_path, pronunce_path]:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            all_data.append(data)
        final_dict = {}

        for i in range(len(all_data[1])):
            for j in range(len(all_data[1][i]["elenco_pronunce"])):
                anno_pronuncia = all_data[1][i]["elenco_pronunce"][j]["anno_pronuncia"]
                if anno_pronuncia not in final_dict.keys():
                    final_dict[anno_pronuncia] = {}
                num_pronuncia = all_data[1][i]["elenco_pronunce"][j]["numero_pronuncia"]
                if num_pronuncia not in final_dict.keys():
                    final_dict[anno_pronuncia][num_pronuncia] = {
                        "testo_pronuncia": None,
                        "testo_massima": None,
                    }
                if final_dict[anno_pronuncia][num_pronuncia]["testo_pronuncia"] is None:
                    final_dict[anno_pronuncia][num_pronuncia]["testo_pronuncia"] = [
                        all_data[1][i]["elenco_pronunce"][j]["testo"]
                    ]
                else:
                    final_dict[anno_pronuncia][num_pronuncia]["testo_pronuncia"].append(
                        all_data[1][i]["elenco_pronunce"][j]["testo"]
                    )

        for i in range(len(all_data[0])):
            for j in range(
                len(all_data[0][i]["corte_costituzionale_archiviomassime"])
            ):  # ! ci sono dei casi in cui non ci sono massime
                try:
                    for k in range(
                        len(
                            all_data[0][i]["corte_costituzionale_archiviomassime"][j][
                                "massime"
                            ]
                        )
                    ):
                        anno_pronuncia = all_data[0][i][
                            "corte_costituzionale_archiviomassime"
                        ][j]["anno_pronuncia"]
                        num_pronuncia = all_data[0][i][
                            "corte_costituzionale_archiviomassime"
                        ][j]["numero_pronuncia"]
                        if anno_pronuncia not in final_dict.keys():
                            final_dict[anno_pronuncia] = {}
                        if num_pronuncia not in final_dict[anno_pronuncia].keys():
                            final_dict[anno_pronuncia][num_pronuncia] = {
                                "testo_pronuncia": None,
                                "testo_massima": None,
                            }
                        if (
                            final_dict[anno_pronuncia][num_pronuncia]["testo_massima"]
                            is None
                        ):
                            final_dict[anno_pronuncia][num_pronuncia][
                                "testo_massima"
                            ] = [
                                all_data[0][i]["corte_costituzionale_archiviomassime"][
                                    j
                                ]["massime"][k]["testo"]
                            ]
                        else:
                            final_dict[anno_pronuncia][num_pronuncia][
                                "testo_massima"
                            ].append(
                                all_data[0][i]["corte_costituzionale_archiviomassime"][
                                    j
                                ]["massime"][k]["testo"]
                            )
                except:
                    pass

        for anno in final_dict.keys():
            for num in final_dict[anno].keys():
                final_dict[anno][num]["testo_massima"] = (
                    list(set(final_dict[anno][num]["testo_massima"]))
                    if final_dict[anno][num]["testo_massima"] is not None
                    else None
                )
        with open(
            destination_path,
            "w",
        ) as json_file:
            json.dump(final_dict, json_file, indent=4)

        # return final_dict


# DATASETS FOR THE SIMPLIFICATION
class MakeDataset_Admin_it(MakeDataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._name = "Admin_it"

    @property
    def name(self) -> str:
        return self._name

    def _group_files_by_number(self, subfolder_name: str):
        file_dict = {}
        file_list = os.path.join(self.root_dir, subfolder_name)
        for file_name in os.listdir(file_list):
            if file_name.endswith(".txt"):
                file_number = "".join(
                    filter(str.isdigit, file_name)
                )  # Extract numbers from filename
                if file_number not in file_dict:
                    file_dict[file_number] = []

                with open(os.path.join(file_list, file_name), "r") as file:
                    content = file.read()
                    file_dict[file_number].append(content)

        return file_dict

    def _save_as_json(self, destination_path: str, file_name: str, file_dict):
        with open(os.path.join(destination_path, f"{file_name}.json"), "w") as file:
            json.dump(file_dict, file, indent=4)

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)
        self._save_as_json(destination_path, "OP", self._group_files_by_number("OP"))
        self._save_as_json(destination_path, "RD", self._group_files_by_number("RD"))
        self._save_as_json(destination_path, "RS", self._group_files_by_number("RS"))


class MakeDataset_Admin_it2(MakeDataset):
    def __init__(self, root_dir: str, file_name: str = "admin-it-l2.txt"):
        self.root_dir = root_dir
        self.file_name = file_name
        self._name = "Admin_it2"

    @property
    def name(self) -> str:
        return self._name

    def _parse_file(self, file_path):
        result = []
        with open(file_path, "r") as file:
            for line in file:
                values = line.strip().split("\t")
                result.append(values)
        return result

    @staticmethod
    def _transform_to_dict(input_list):
        list_of_dicts = [
            {
                "original": sublist[0],
                "simplification": sublist[1],
                "l2_semp": sublist[2],
            }
            for sublist in input_list
        ]
        return list_of_dicts

    def _save_as_json(self, destination_path: str, file_name: str, file_dict):
        with open(os.path.join(destination_path, file_name), "w") as file:
            json.dump(file_dict, file, indent=4)

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)
        result_l2 = self._parse_file(os.path.join(self.root_dir, self.file_name))
        result_l2_list_of_dict = self._transform_to_dict(result_l2)
        self._save_as_json(destination_path, "admin-it-l2.json", result_l2_list_of_dict)


class MakeDataset_OneStopEnglishCorpus(MakeDataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._name = "OneStopEnglishCorpus"

    @property
    def name(self) -> str:
        return self._name

    def _read_files_in_folders(self):
        data_dict = {}

        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(folder_path, file_name)
                        # Extract the key from file name
                        key = file_name.split("-")[0]
                        for enc in ENCODINGS:
                            try:
                                with open(file_path, "r", encoding=enc) as file:
                                    content = file.read().strip()

                                    if key not in data_dict:
                                        data_dict[key] = [content]
                                    else:
                                        data_dict[key].append(content)
                            except:
                                print(
                                    f"Error reading file: {file_path} with encoding: {enc}"
                                )
        return data_dict

    def _save_as_json(self, destination_path: str, file_name: str, file_dict):
        with open(os.path.join(destination_path, file_name), "w") as file:
            json.dump(file_dict, file, indent=4)

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)
        data_list_dict = self._read_files_in_folders()
        self._save_as_json(
            destination_path, "onestopenglishcorpus.json", data_list_dict
        )


class MakeDataset_Paccss_it(MakeDataset):
    def __init__(self, root_dir: str, file_name: str = "PACCSS-IT.txt"):
        self.root_dir = root_dir
        self.file_name = file_name
        self._name = "Paccss_it"

    @property
    def name(self) -> str:
        return self._name

    def _create_dataframe_from_file(self):
        file_path = os.path.join(self.root_dir, self.file_name)

        with open(file_path, "r") as file:
            lines = file.readlines()
            column_names = lines[0].strip().split("\t")

        # Use the rest of the lines to create the DataFrame
        data = [line.strip().split("\t") for line in lines[1:]]

        # Create the DataFrame with the extracted column names
        data_list_dict = pd.DataFrame(data, columns=column_names).to_dict(
            orient="records"
        )
        return data_list_dict

    def _save_as_json(self, destination_path: str, file_name: str, file_dict):
        with open(os.path.join(destination_path, file_name), "w") as file:
            json.dump(file_dict, file, indent=4)

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)
        data_list_dict = self._create_dataframe_from_file()
        self._save_as_json(destination_path, f"paccss_it.json", data_list_dict)


class MakeDataset_Simpa(MakeDataset):
    def __init__(
        self,
        root_dir: str,
        ls_original: str = "ls.original",
        ls_simplified: str = "ls.simplified",
        ss_ls_simplified: str = "ss.ls-simplified",
        ss_original: str = "ss.original",
        ss_simplified: str = "ss.simplified",
    ):
        self.root_dir = root_dir
        self.ls_original = ls_original
        self.ls_simplified = ls_simplified
        self.ss_ls_simplified = ss_ls_simplified
        self.ss_original = ss_original
        self.ss_simplified = ss_simplified
        self._name = "Simpa"

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _read_file(file_path):
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
                return lines
        except FileNotFoundError:
            print(f"File not found. Please check the file path {file_path}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def _union_lists(*lists):
        result = []
        for i in range(len(lists[0])):
            sublist = []
            for lst in lists:
                sublist.append(lst[i])
            result.append(sublist)
        return result

    def _save_as_json(self, destination_path: str, file_name: str, file_dict):
        with open(os.path.join(destination_path, file_name), "w") as file:
            json.dump(file_dict, file, indent=4)

    @staticmethod
    def transform_to_dict_for_three(input_list):
        list_of_dicts = [
            {
                "original": sublist[0],
                "simplification1": sublist[1],
                "simplification2": sublist[2],
            }
            for sublist in input_list
        ]
        return list_of_dicts

    @staticmethod
    def transform_to_dict_for_two(input_list):
        list_of_dicts = [
            {"original": sublist[0], "simplification": sublist[1]}
            for sublist in input_list
        ]
        return list_of_dicts

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)
        ls_original_data = self._read_file(
            os.path.join(self.root_dir, self.ls_original)
        )
        ls_similified_data = self._read_file(
            os.path.join(self.root_dir, self.ls_simplified)
        )
        ss_ls_similified_data = self._read_file(
            os.path.join(self.root_dir, self.ss_ls_simplified)
        )
        ss_original_data = self._read_file(
            os.path.join(self.root_dir, self.ss_original)
        )
        ss_similified_data = self._read_file(
            os.path.join(self.root_dir, self.ss_simplified)
        )

        ls_original_ls_similified = self._union_lists(
            ls_original_data, ls_similified_data
        )

        ss_ls_similified_ss_original_ss_similified = self._union_lists(
            ss_ls_similified_data, ss_original_data, ss_similified_data
        )

        self._save_as_json(
            destination_path,
            "ss_ls_similified_ss_original_ss_similified.json",
            self.transform_to_dict_for_three(
                ss_ls_similified_ss_original_ss_similified
            ),
        )

        self._save_as_json(
            destination_path,
            "ls_original_ls_similified.json",
            self.transform_to_dict_for_two(ls_original_ls_similified),
        )


class MakeDataset_Simpitiki(MakeDataset):
    def __init__(self, root_dir: str, file_name: str = "simpitiki-v2.xml"):
        self.root_dir = root_dir
        self.file_name = file_name
        self._name = "Simpitiki"
        self._conversion_type_dict = {
            "1": "Split",
            "2": "Merge",
            "3": "Reordering",
            "11": "Insert - Verb",
            "12": "Insert - Subject",
            "13": "Insert - Other",
            "21": "Delete - Verb",
            "22": "Delete - Subject",
            "23": "Delete - Other",
            "31": "Transformation - Lexical Substitution (word level)",
            "32": "Transformation - Lexical Substitution (phrase level)",
            "33": "Transformation - Anaphoric replacement",
            "34": "Transformation - Noun to Verb",
            "35": "Transformation - Verb to Noun (nominalization)",
            "36": "Transformation - Verbal Voice",
            "37": "Transformation - Verbal Features",
        }

    @property
    def conversion_type_dict(self) -> dict:
        return self._conversion_type_dict

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _parse_xml_file(file_path):
        result = []

        tree = ET.parse(file_path)
        root = tree.getroot()

        for simplification in root.findall(".//simplification"):
            simplification_data = {}
            simplification_data["type"] = simplification.get("type")
            simplification_data["origin"] = simplification.get("origin")

            before_text = simplification.find("before").text
            simplification_data["before"] = (
                before_text.strip() if before_text is not None else ""
            )

            after_text = simplification.find("after").text
            simplification_data["after"] = (
                after_text.strip() if after_text is not None else ""
            )

            result.append(simplification_data)

        return result

    @staticmethod
    def _replace_dict_value(dict1, dict2, key):
        if dict1[key] in dict2.keys():
            dict1[key] = dict2[dict1[key]]
        return dict1

    def _replace_lst_of_dict_value(self, list_of_dict, dict2, key):
        return [self._replace_dict_value(ld, dict2, key) for ld in list_of_dict]

    def _save_as_json(self, destination_path: str, file_name: str, file_dict):
        with open(os.path.join(destination_path, file_name), "w") as file:
            json.dump(file_dict, file, indent=4)

    def create_data(self, destination_path: str):
        create_folder_if_not_exists(destination_path)
        parsed_data = self._parse_xml_file(os.path.join(self.root_dir, self.file_name))
        final_simpliwoki_dict = self._replace_lst_of_dict_value(
            parsed_data, self._conversion_type_dict, "type"
        )
        self._save_as_json(destination_path, "simpitiki-v2.json", final_simpliwoki_dict)
