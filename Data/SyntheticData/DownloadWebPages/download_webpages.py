import os
import shutil
from bs4 import BeautifulSoup
from Config.setting import *
from zipfile import ZipFile
import time
from selenium import webdriver
from tqdm import tqdm
import requests
import json
import numpy as np


def save_to_json(dictionary, file_path):
    json_object = json.dumps(dictionary, indent=4)
    with open(file_path, mode='w', encoding='utf-8') as data_file_writer:
        data_file_writer.write(json_object)


class WebTemplateDownloader:
    TEMPLATEMO_URL = r'https://templatemo.com'
    FREE_CSS_URL = r'https://www.free-css.com'
    PLATFORMS_FREE_TEMPLATES = ['templatemo', 'free_css']

    def __init__(self, downloading_name='templates', platforms_to_use='all'):
        self.downloading_name = downloading_name
        self.dst_folder = os.path.join(OFFLINE_WEB_PATH, downloading_name)
        os.makedirs(self.dst_folder, exist_ok=True)
        if platforms_to_use == 'all':
            self.platforms_to_use = self.PLATFORMS_FREE_TEMPLATES
        else:
            self.platforms_to_use = platforms_to_use
        self.platforms_to_folders = {}
        for platform in self.platforms_to_use:
            platform_folder = os.path.join(self.dst_folder, platform)
            os.makedirs(platform_folder, exist_ok=True)
            self.platforms_to_folders[platform] = platform_folder

    def download(self):
        links = {}
        offline_paths = {}
        for platform in self.platforms_to_use:
            if platform == 'templatemo':
                links_templatemo, main_offline_pages_templatemo = self.download_templatemo()
                links['templatemo'] = links_templatemo
                offline_paths['templatemo'] = main_offline_pages_templatemo
            elif platform == 'free_css':
                links_free_css, main_offline_pages_free_css = self.download_free_css()
                links['free_css'] = links_free_css
                offline_paths['free_css'] = main_offline_pages_free_css
        save_to_json(links, os.path.join(self.dst_folder, 'links.json'))
        save_to_json(offline_paths, os.path.join(WEB_LINKS_PATH, '{}.json'.format(self.downloading_name)))

    def download_templatemo(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_experimental_option("prefs", {"download.default_directory": self.platforms_to_folders['templatemo']})
        driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=options)
        links = []
        for page in range(1, 51):
            page_url = self.TEMPLATEMO_URL + '/page/{}'.format(page)
            page_src = requests.get(page_url).text
            page_src = BeautifulSoup(page_src, features="html.parser")
            links += [self.TEMPLATEMO_URL + item['href'] for item in page_src.find_all('a') if '/tm' == item['href'][:3]]
        links = list(set(links))
        for link in tqdm(links):
            try:
                driver.get(link)
                el = driver.find_element('id', 'tm-download')
                driver.execute_script("window.scrollTo(0, {});".format(el.location['y']))
                el.click()
            except Exception as e:
                print(str(e))
        time.sleep(20)
        downloaded_zip_files = [os.path.join(self.platforms_to_folders['templatemo'], fn) for fn in os.listdir(self.platforms_to_folders['templatemo']) if fn.split('.')[-1].lower() == 'zip']
        for downloaded_zip_file in downloaded_zip_files:
            with ZipFile(downloaded_zip_file, 'r') as zObject:
                zObject.extractall(path=self.platforms_to_folders['templatemo'])
            os.remove(downloaded_zip_file)
        main_offline_pages = []
        for fn in os.listdir(self.platforms_to_folders['templatemo']):
            fp = os.path.join(self.platforms_to_folders['templatemo'], fn)
            if not os.path.isdir(fp):
                os.remove(fp)
            elif not os.path.isfile(os.path.join(fp, 'index.html')):
                shutil.rmtree(fp)
            else:
                main_offline_pages.append(os.path.join(fp, 'index.html'))
        return links, main_offline_pages

    def download_free_css(self):
        num_templates = 0
        former_num_templates = -1
        links = []
        while num_templates > former_num_templates:
            success = False
            while not success:
                try:
                    time.sleep(10)
                    former_num_templates = num_templates
                    page_url = self.FREE_CSS_URL + '/free-css-templates?start={}'.format(num_templates)
                    page_src = requests.get(page_url).text
                    page_src = BeautifulSoup(page_src, features="html.parser")
                    div_urls = page_src.find('div', {'id': 'showcase'})
                    urls = [self.FREE_CSS_URL + a['href'] for a in div_urls.find_all('a')]
                    links += urls
                    num_templates += len(urls)
                    print(num_templates)
                    success = True
                except Exception as e:
                    print(str(e))
                    print('Sleeping...')
                    time.sleep(120)
        links = list(set(links))
        downloaded_zip_files = []
        for link in tqdm(links):
            success = False
            retries = 3
            while not success and retries > 0:
                try:
                    time.sleep(10)
                    page_src = requests.get(link).text
                    page_src = BeautifulSoup(page_src, features="html.parser")
                    download_item = page_src.find('li', {'class': 'dld'})
                    zip_link = self.FREE_CSS_URL + download_item.find('a')['href']
                    downloaded_zip_path = os.path.join(self.platforms_to_folders['free_css'], zip_link.split(os.sep)[-1])
                    if not os.path.isfile(downloaded_zip_path):
                        response = requests.get(zip_link)
                        open(downloaded_zip_path, "wb").write(response.content)
                        downloaded_zip_files.append(downloaded_zip_path)
                    else:
                        downloaded_zip_files.append(downloaded_zip_path)
                        print('Already exists: {}'.format(downloaded_zip_path))
                    success = True
                except Exception as e:
                    print(str(e))
                    print('Sleeping...')
                    time.sleep(120)
                    retries -= 1
        downloaded_zip_files = list(set(downloaded_zip_files))
        for downloaded_zip_file in downloaded_zip_files:
            with ZipFile(downloaded_zip_file, 'r') as zObject:
                zObject.extractall(path=self.platforms_to_folders['free_css'])
            os.remove(downloaded_zip_file)
        main_offline_pages = []
        for fn in os.listdir(self.platforms_to_folders['free_css']):
            fp = os.path.join(self.platforms_to_folders['free_css'], fn)
            if not os.path.isdir(fp):
                os.remove(fp)
            elif not os.path.isfile(os.path.join(fp, 'index.html')):
                shutil.rmtree(fp)
            else:
                main_offline_pages.append(os.path.join(fp, 'index.html'))
        return links, main_offline_pages



class WikiDownloader:

    def __init__(self, random_depth_each_result=14, random_breath_each_result=2, starting_page=r'https://en.wikipedia.org/wiki/Main_Page'):
        self.wiki_name = starting_page.split('/')[2]
        self.dst_folder = os.path.join(OFFLINE_WEB_PATH, 'wiki_tmp')
        os.makedirs(self.dst_folder, exist_ok=True)
        self.starting_page = starting_page
        self.random_depth_each_result = random_depth_each_result
        self.random_breath_each_result = random_breath_each_result

    def __dfs(self, link, depth=0):
        links = []
        if depth >= self.random_depth_each_result:
            return links
        response = requests.get(link)
        soup = BeautifulSoup(response.text, "html.parser")
        webpage_links = soup.findAll('a')
        webpage_links = [i for i in webpage_links if 'href' in i.attrs.keys() and i.attrs['href'][:6] == r'/wiki/']
        breath_counter = 0
        while breath_counter < self.random_breath_each_result and len(webpage_links) > 0:
            sample = np.random.randint(0, len(webpage_links))
            random_link = webpage_links[sample]['href']
            random_link = r'https://en.wikipedia.org' + random_link
            links.append(random_link)
            webpage_links.pop(sample)
            links += self.__dfs(random_link, depth=depth + 1)
            breath_counter += 1
        return links

    def __downlowad_links(self, links):
        cwd = os.getcwd()
        os.chdir(self.dst_folder)
        try:
            for link in links:
                os.system("wget -r --convert-links {} -T 5".format(link))
        except Exception as e:
            print(str(e))
        os.chdir(cwd)

    def download(self):
        web_links = []
        links = self.__dfs(self.starting_page)
        self.__downlowad_links(links)
        for (parent_dir, _, file_names) in os.walk(self.dst_folder):
            for file_name in file_names:
                file_path = os.path.join(parent_dir, file_name)
                if os.path.isfile(file_path):
                    os.rename(file_path, file_path + '.html')
                    web_links.append(file_path + '.html')
        shutil.move(os.path.join(self.dst_folder, self.wiki_name, 'wiki'), os.path.join(OFFLINE_WEB_PATH, 'wiki'))
        shutil.rmtree(self.dst_folder)
        return web_links


if __name__ == '__main__':
    wiki = WikiDownloader()
    wiki.download()
