import os
import json
from uuid import uuid4
from Config.setting import *
from collections.abc import Iterable


def print_list(arr, depth=0):
    for v in arr:
        if type(v) == dict:
            print_dictionary(v, depth + 1)
        elif type(v) != str and isinstance(v, Iterable):
            print_list(v, depth + 1)
        else:
            print('\t' * (depth + 1) + str(v))


def print_dictionary(dictionary, depth=0):
    for k, v in dictionary.items():
        print('\t' * depth + k)
        if type(v) == dict:
            print_dictionary(v, depth + 1)
        elif type(v) != str and isinstance(v, Iterable):
            print_list(v, depth + 1)
        else:
            print('\t' * (depth + 1) + str(v))


def make_tmp_folder():
    tmp_folder = os.path.join(TMP_PATH, 'a' + str(uuid4()))
    os.makedirs(tmp_folder, exist_ok=True)
    return tmp_folder


def save_to_json(dictionary, file_path):
    json_object = json.dumps(dictionary, indent=4)
    with open(file_path, mode='w', encoding='utf-8') as data_file_writer:
        data_file_writer.write(json_object)


def load_json(file_path):
    data = json.load(open(file_path, mode='r', encoding='utf-8'))
    return data


def save_links(links_file_name: str, links: list):
    save_to_json({'links': links}, os.path.join(WEB_LINKS_PATH, '{}.json'.format(links_file_name)))

