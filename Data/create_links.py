import os, json
from Config.setting import *
from GeneralUtils.utils import save_links


def create_links():
    all_links = []
    for offline_pages_type in os.listdir(OFFLINE_WEB_PATH):
        offline_pages_type_folder = os.path.join(OFFLINE_WEB_PATH, offline_pages_type)
        links = _search_files(offline_pages_type_folder)
        if len(links) > 0:
            all_links += links
            save_links(offline_pages_type, links)
    save_links('all_offline_webpage', all_links)


def _search_files(folder_path, file_names='all', depth=2):
    links = []
    if os.path.isdir(folder_path):
        for fn in os.listdir(folder_path):
            fp = os.path.join(folder_path, fn)
            if os.path.isfile(fp) and fp.split('.')[-1].lower() in ('html', 'htm', 'mhtml') and (file_names == 'all' or fn.split('.')[0] in file_names):
                links.append(fp)
            elif os.path.isdir(fp) and depth > 0:
                links += _search_files(fp, file_names=('index',), depth=depth - 1)
    return links

