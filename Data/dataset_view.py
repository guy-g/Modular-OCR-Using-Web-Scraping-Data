import os
import json
from collections import Counter
from Data.language import Language
from tqdm import tqdm
import emoji
from GeneralUtils.utils import *
from Config.setting import *


class OCRDatasetView:
    IMAGES_EXT = ('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif')
    DATASET_EXT = ('png', 'jpg')
    '''
    Creating or Loading a DatasetView
    '''

    def __init__(self, dataset_view_name, datasets_views_folder, datasets_folder,
                 load_view_file=False, inference=False, language_name=None,
                 language_folder=None, max_word_length=None,
                 running_names='all', detection=True, recognition=True, layout=True,
                 to_save=True, enable_icon_or_emoji_text=False, ignore_direction=False):
        self.dataset_view_name = dataset_view_name
        self.datasets_views_folder = datasets_views_folder
        self.datasets_folder = datasets_folder
        self.dataset_view_file = os.path.join(datasets_views_folder, dataset_view_name + '.json')
        self.inference = inference
        if not load_view_file:
            self.language_name = language_name
            self.language_folder = language_folder
            if language_name is not None and language_folder is not None:
                self.language = Language(language_name, language_folder, True)
            else:
                self.language = None
            self.max_word_length = max_word_length
            self.detection = detection
            self.recognition = recognition
            self.layout = layout
            self.enable_icon_or_emoji_text = enable_icon_or_emoji_text
            self.ignore_direction = ignore_direction
            if running_names != 'all':
                self.running_folders = [os.path.join(datasets_folder, folder_name) for folder_name in os.listdir(datasets_folder) if (os.path.isdir(os.path.join(datasets_folder, folder_name)) and (folder_name in running_names))]
            else:
                self.running_folders = [datasets_folder]
            self.image_paths_to_detection_data = {}
            self.image_paths_to_recognition_data = {}
            self.image_paths_to_layout_data = {}
            self.num_pixels_each_class = {
                'detection': Counter(),
                'layout': Counter()
            }
            if detection:
                self.image_paths_to_detection_data = self.__get_data_detection()
            else:
                self.num_pixels_each_class['detection'] = [0, 0]
            if recognition:
                self.image_paths_to_recognition_data = self.__get_data_recognition()
            if layout:
                self.image_paths_to_layout_data = self.__get_data_layout()
            else:
                self.num_pixels_each_class['layout'] = [0, 0]
            if to_save:
                self.__save()
        else:
            self.__load()

    def __get_data_recognition(self):
        image_paths_to_recognition_data = {}
        # debug = 0
        for running_folder in self.running_folders:
            for (parent_dir, _, file_names) in tqdm(os.walk(running_folder)):
                if (not self.inference) and parent_dir.split(os.sep)[-1] == 'recognition_data' and 'data.json' in file_names:
                        # and os.path.isdir(os.path.join(f'{os.sep}'.join(parent_dir.split(os.sep)[:-1]), 'layout_data')):
                    recognition_json = os.path.join(parent_dir, 'data.json')
                    try:
                        data = json.load(open(recognition_json, mode='r', encoding='utf-8'))
                    except:
                        continue
                    for (file_name, file_data) in tqdm(data.items()):
                        if self.language is not None \
                                and (self.ignore_direction or (str(self.language.direction) == str(file_data['direction']))) \
                                and (self.max_word_length is None or len(file_data['text']) <= self.max_word_length) \
                                and (self.enable_icon_or_emoji_text or len(emoji.emoji_list(file_data['text'])) == 0):
                            if not self.language.is_accept(file_data['text']):
                                continue
                            else:
                                if os.path.isfile(os.path.join(parent_dir, file_name)):
                                    # debug += 1
                                    # continue
                                    image_paths_to_recognition_data[os.path.join(parent_dir, file_name)] = file_data
                                    image_paths_to_recognition_data[os.path.join(parent_dir, file_name)]['text'] = self.language.normalize_word(file_data['text'])
                        elif self.language is None and os.path.isfile(os.path.join(parent_dir, file_name)) and (self.max_word_length is None or len(file_data['text']) <= self.max_word_length) and (self.enable_icon_or_emoji_text or len(emoji.emoji_list(file_data['text'])) == 0):
                            # debug += 1
                            # continue
                            image_paths_to_recognition_data[os.path.join(parent_dir, file_name)] = file_data
                        else:
                            print(file_data['text'])
                elif self.inference:
                    for file_name in file_names:
                        if file_name.lower().endswith(self.IMAGES_EXT):
                            image_paths_to_recognition_data[os.path.join(parent_dir, file_name)] = {
                                'text': None,
                                'image_size': None,
                                'rotated': None,
                                'mask_file': None,
                                'direction': None
                            }
        # save_to_json({'dataset size': debug}, os.path.join(self.datasets_views_folder, self.dataset_view_name + '_dataset_size.json'))
        return image_paths_to_recognition_data

    def __get_data_detection(self):
        image_paths_to_detection_data = {}
        max_idx = -1
        for running_folder in self.running_folders:
            for (parent_dir, _, file_names) in tqdm(os.walk(running_folder)):
                if not self.inference:
                    for ext in OCRDatasetView.DATASET_EXT:
                        if parent_dir.split(os.sep)[-1] == 'detection_data' and 'data.json' in file_names and 'image.' + ext in file_names:
                            file_name = 'image.' + ext
                            detection_json = os.path.join(parent_dir, 'data.json')
                            try:
                                data = json.load(open(detection_json, mode='r', encoding='utf-8'))
                            except:
                                continue
                            icons = False
                            if not self.enable_icon_or_emoji_text:
                                for w in data['tags_to_text'].values():
                                    if len(emoji.emoji_list(w)) > 0:
                                        icons = True
                            if not icons:
                                image_paths_to_detection_data[os.path.join(parent_dir, file_name)] = {
                                    'bounding_boxes_text_mask_file': os.path.join(parent_dir, 'bounding_boxes_words_mask.png'),
                                    'bounding_boxes_text_mask_transformed_file': os.path.join(parent_dir, 'bounding_boxes_words_mask_transformed.png'),
                                    'bounding_boxes_image_mask_file': os.path.join(parent_dir, 'bounding_boxes_images_mask.png'),
                                    'bounding_boxes_image_mask_transformed_file': os.path.join(parent_dir, 'bounding_boxes_images_mask_transformed.png'),
                                    'mask_file': os.path.join(parent_dir, 'mask.png'),
                                    'background_file': os.path.join(parent_dir, 'background.' + ext),
                                    'mask_images_file': os.path.join(parent_dir, 'mask_images.png'),
                                    'background_images_file': os.path.join(parent_dir, 'background_images.' + ext),
                                    'elastic': os.path.join(parent_dir, 'indices.npy'),
                                    'data_file': detection_json,
                                    'image_size': data['image_size'],
                                    'url': data['url'],
                                    'num_words': len(data['tags_to_text'].keys()),
                                    'background_changed': data['background_changed']
                                }
                                num_pixels = float(sum(data['num_pixels_each_class']))
                                for idx in range(len(data['num_pixels_each_class'])):
                                    self.num_pixels_each_class['detection'][idx] += data['num_pixels_each_class'][idx] / num_pixels
                                    if idx > max_idx:
                                        max_idx = idx
                elif self.inference:
                    for file_name in file_names:
                        if file_name.lower().endswith(self.IMAGES_EXT):
                            data = {
                                'bounding_boxes_text_mask_file': None,
                                'bounding_boxes_text_mask_transformed_file': None,
                                'bounding_boxes_image_mask_file': None,
                                'bounding_boxes_image_mask_transformed_file': None,
                                'background_file': None,
                                'mask_file': None,
                                'image_size': None,
                                'data_file': None,
                                'bounding_boxes_text': None,
                                'bounding_boxes_and_text_for_recognition': None,
                                'bounding_boxes_images': None,
                                'bounding_boxes_images_not_overlapping_text': None,
                                'mask_images_file': None,
                                'background_images_file': None,
                                'num_pixels_each_class': None,
                                'perspective': None,
                                'elastic': None,
                                'url': None,
                                'num_words': None,
                                'background_changed': None
                            }
                            image_paths_to_detection_data[os.path.join(parent_dir, file_name)] = data
        self.num_pixels_each_class['detection'] = [self.num_pixels_each_class['detection'][idx] for idx in range(max_idx + 1)]
        return image_paths_to_detection_data

    def __get_data_layout(self):
        image_paths_to_layout_data = {}
        for running_folder in self.running_folders:
            for (parent_dir, _, file_names) in tqdm(os.walk(running_folder)):
                if not self.inference:
                    for ext in OCRDatasetView.DATASET_EXT:
                        if parent_dir.split(os.sep)[-1] == 'layout_data' and 'data.json' in file_names and 'image.' + ext in file_names:
                            file_name = 'image.' + ext
                            layout_json = os.path.join(parent_dir, 'data.json')
                            try:
                                data = json.load(open(layout_json, mode='r', encoding='utf-8'))
                            except:
                                continue
                            image_paths_to_layout_data[os.path.join(parent_dir, file_name)] = {
                                'paragraphs': data['paragraphs']
                            }
                elif self.inference:
                    for file_name in file_names:
                        if file_name.lower().endswith(self.IMAGES_EXT):
                            data = {
                                'paragraphs': None
                            }
                            image_paths_to_layout_data[os.path.join(parent_dir, file_name)] = data
        return image_paths_to_layout_data

    def __save(self):
        data = {
            'dataset_view_name': self.dataset_view_name,
            'datasets_views_folder': self.datasets_views_folder,
            'datasets_folder': self.datasets_folder,
            'dataset_view_file': self.dataset_view_file,
            'inference': self.inference,
            'language_name': self.language_name,
            'language_folder': self.language_folder,
            'max_word_length': self.max_word_length,
            'detection': self.detection,
            'recognition': self.recognition,
            'layout': self.layout,
            'running_folders': self.running_folders,
            'image_paths_to_detection_data': self.image_paths_to_detection_data,
            'image_paths_to_recognition_data': self.image_paths_to_recognition_data,
            'image_paths_to_layout_data': self.image_paths_to_layout_data,
            'num_pixels_each_class': self.num_pixels_each_class,
            'enable_icon_or_emoji_text': self.enable_icon_or_emoji_text,
            'ignore_direction': self.ignore_direction
        }
        # write_dictionary_by_chunks(data, open(self.dataset_view_file, mode='w', encoding='utf-8'))
        save_to_json(data, self.dataset_view_file)

    def __load(self):
        data = json.load(open(self.dataset_view_file, mode='r', encoding='utf-8'))
        self.dataset_view_name = data['dataset_view_name']
        self.datasets_views_folder = data['datasets_views_folder']
        self.datasets_folder = data['datasets_folder']
        self.inference = data['inference']
        self.language_name = data['language_name']
        self.language_folder = data['language_folder']
        self.max_word_length = data['max_word_length']
        self.detection = data['detection']
        self.recognition = data['recognition']
        self.layout = data['layout']
        self.running_folders = data['running_folders']
        self.image_paths_to_detection_data = data['image_paths_to_detection_data']
        self.image_paths_to_recognition_data = data['image_paths_to_recognition_data']
        self.image_paths_to_layout_data = data['image_paths_to_layout_data']
        self.num_pixels_each_class = data['num_pixels_each_class']
        self.enable_icon_or_emoji_text = data['enable_icon_or_emoji_text']
        self.ignore_direction = data['ignore_direction']
        if self.language_name is not None and self.language_folder is not None:
            self.language = Language(self.language_name, self.language_folder, True)
        else:
            self.language = None


class LightOCRDatasetView:
    IMAGES_EXT = ('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif')
    DATASET_EXT = ('png', 'jpg')
    '''
    Creating or Loading a DatasetView
    '''

    def __init__(self, dataset_view_name, running_names='all', detection=True, recognition=True, to_save=True):
        self.dataset_view_name = dataset_view_name
        self.dataset_view_file = os.path.join(DATASETS_VIEWS_PATH, dataset_view_name + '.json')
        if not to_save or not os.path.isfile(self.dataset_view_file):
            self.image_paths_to_detection_data = {}
            self.image_paths_to_recognition_data = {}
            self.num_pixels_each_class = {'detection': [0, 0]}
            dataset_names = running_names if running_names != 'all' else [fn for fn in os.listdir(DATASETS_PATH) if os.path.isdir(DATASETS_PATH, fn)]
            if detection:
                detection_dirs = [os.path.join(DATASETS_PATH, dataset_name, fn, 'detection_data') for dataset_name in dataset_names for fn in os.listdir(os.path.join(DATASETS_PATH, dataset_name))]
                detection_dirs = [d for d in detection_dirs if os.path.isdir(d)]
                for detection_dir in detection_dirs:
                    if os.path.isfile(os.path.join(detection_dir, 'data.json')):
                        ext = 'png' if 'image.png' in os.listdir(detection_dir) else 'jpg'
                        data = json.load(open(os.path.join(detection_dir, 'data.json'), mode='r', encoding='utf-8'))
                        self.image_paths_to_detection_data[os.path.join(detection_dir, 'image.{}'.format(ext))] = {
                            'data_file': os.path.join(detection_dir, 'data.json'),
                            'url': data['url']
                        }
                        num_pixels_each_class = data['num_pixels_each_class']
                        self.num_pixels_each_class['detection'][0] += num_pixels_each_class[0]
                        self.num_pixels_each_class['detection'][1] += num_pixels_each_class[1]
            if recognition:
                recognition_dirs = [os.path.join(DATASETS_PATH, dataset_name, fn, 'recognition_data') for dataset_name in dataset_names for fn in os.listdir(os.path.join(DATASETS_PATH, dataset_name))] + \
                                   [os.path.join(DATASETS_PATH, dataset_name, 'recognition_data') for dataset_name in dataset_names]
                recognition_dirs = [d for d in recognition_dirs if os.path.isdir(d)]
                for recognition_dir in recognition_dirs:
                    # recognition_dir_data = os.path.join(recognition_dir, 'data.json')
                    # if os.path.isfile(recognition_dir_data):
                    for fn in os.listdir(recognition_dir):
                        if fn.lower().endswith(OCRDatasetView.IMAGES_EXT) and 'mask' not in fn:
                            self.image_paths_to_recognition_data[os.path.join(recognition_dir, fn)] = {}
            if to_save:
                self.__save()
        else:
            self.__load()

    def __save(self):
        data = {
            'image_paths_to_detection_data': self.image_paths_to_detection_data,
            'image_paths_to_recognition_data': self.image_paths_to_recognition_data,
            'num_pixels_each_class': self.num_pixels_each_class
        }
        save_to_json(data, self.dataset_view_file)

    def __load(self):
        data = json.load(open(self.dataset_view_file, mode='r', encoding='utf-8'))
        self.image_paths_to_detection_data = data['image_paths_to_detection_data']
        self.image_paths_to_recognition_data = data['image_paths_to_recognition_data']
        self.num_pixels_each_class = data['num_pixels_each_class']


def recognition_data_generator(running_names='all', batch_size=1000000):
    image_paths_to_recognition_data = {}
    dataset_names = running_names if running_names != 'all' else [fn for fn in os.listdir(DATASETS_PATH) if os.path.isdir(DATASETS_PATH, fn)]
    recognition_dirs = [os.path.join(DATASETS_PATH, dataset_name, fn, 'recognition_data') for dataset_name in dataset_names for fn in os.listdir(os.path.join(DATASETS_PATH, dataset_name))] + \
                       [os.path.join(DATASETS_PATH, dataset_name, 'recognition_data') for dataset_name in dataset_names]
    # recognition_dirs = [d for d in recognition_dirs if os.path.isdir(d)]
    for recognition_dir in tqdm(recognition_dirs):
        recognition_dir_data = os.path.join(recognition_dir, 'data.json')
        if os.path.isfile(recognition_dir_data):
            recognition_dir_data = json.load(open(recognition_dir_data, mode='r', encoding='utf-8'))
            for fn in os.listdir(recognition_dir):
                if fn.lower().endswith(OCRDatasetView.IMAGES_EXT) and 'mask' not in fn:
                    if fn in recognition_dir_data.keys():
                        image_paths_to_recognition_data[os.path.join(recognition_dir, fn)] = recognition_dir_data[fn]
                if len(image_paths_to_recognition_data.keys()) >= batch_size:
                    yield image_paths_to_recognition_data
                    image_paths_to_recognition_data = {}
    if len(image_paths_to_recognition_data.keys()) > 0:
        yield image_paths_to_recognition_data

