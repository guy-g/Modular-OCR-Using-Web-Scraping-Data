import os
import shutil
import json
import sys
import time
import uuid
from imgaug.augmentables.segmaps import SegmentationMapOnImage, SegmentationMapsOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import numpy as np
import cv2
from Data.dataset_view import OCRDatasetView
from PIL import Image
from Data.Utils.write_json import *
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from Config.setting import NATURAL_BACKGROUNDS_PATH, TEXTURE_BACKGROUNDS_PATH
from GeneralUtils.draw_demonstration import draw_images_paths_to_bounding_boxes_and_text, draw_images_paths_to_bounding_boxes
from Config.setting import FONTS_PATH
from Data.conversion_to_coco import *
from Config.setting import *
from Data.Munit.munit import munit_inference, INFERENCE_TYPES
from GeneralUtils.utils import *


class DatasetAugmentation:
    '''
    DatasetView -> Create Augmented Dataset+DatasetView
    '''
    def __init__(self, 
                 augmented_dataset_name,
                 log,
                 ocr_dataset_view: OCRDatasetView,
                 transformations_to_probabilities_detection=None,
                 transformations_to_probabilities_recognition=None,
                 transformations_to_probabilities_layout=None,
                 augment_recognition_by_detection=True,
                 style_transfer_model_names_to_probabilities=None,
                 style_transfer_model_names_to_style_transfer_type=None,
                 gpu_ids='0',
                 demonstration_font_name='arial',
                 salt_and_pepper_prob=0.05,
                 rectangular_bounding_boxes=True,
                 detection_data_classes_field_names=('bounding_boxes_text',),
                 detection_data_fields_to_mask_in_loss=('bounding_boxes_images',),
                 consider_elastic=True,
                 consider_perspective=True,
                 style_transfer_only_on_clean_background=False,
                 enable_icon_or_emoji_text=False,
                 imgaug_transformations_detection=None,
                 imgaug_transformations_recognition=None,
                 imgaug_transformations_layout=None,
                 ):
        self.augmented_dataset_name = augmented_dataset_name
        self.augmented_dataset_folder = os.path.join(DATASETS_PATH, augmented_dataset_name)
        os.makedirs(self.augmented_dataset_folder, exist_ok=True)
        self.gpu_ids = gpu_ids
        self.image_paths_to_recognition_data = ocr_dataset_view.image_paths_to_recognition_data
        self.image_paths_to_detection_data = ocr_dataset_view.image_paths_to_detection_data
        self.image_paths_to_layout_data = ocr_dataset_view.image_paths_to_layout_data
        self.log = log
        self.demonstration_font_name = demonstration_font_name
        self.salt_and_pepper_prob = salt_and_pepper_prob
        self.rectangular_bounding_boxes = rectangular_bounding_boxes
        self.detection_data_classes_field_names = detection_data_classes_field_names
        self.detection_data_fields_to_mask_in_loss = detection_data_fields_to_mask_in_loss
        self.consider_elastic = consider_elastic
        self.consider_perspective = consider_perspective
        self.style_transfer_only_on_clean_background = style_transfer_only_on_clean_background
        self.enable_icon_or_emoji_text = enable_icon_or_emoji_text
        self.transformations_to_probabilities_detection = transformations_to_probabilities_detection if transformations_to_probabilities_detection is not None else {}
        self.transformations_to_probabilities_recognition = transformations_to_probabilities_recognition if transformations_to_probabilities_recognition is not None else {}
        self.transformations_to_probabilities_layout = transformations_to_probabilities_layout if transformations_to_probabilities_layout is not None else {}
        self.augment_recognition_by_detection = augment_recognition_by_detection
        self.style_transfer_model_names_to_probabilities = style_transfer_model_names_to_probabilities if style_transfer_model_names_to_probabilities is not None else {}
        self.style_transfer_model_names_to_style_transfer_type = style_transfer_model_names_to_style_transfer_type
        self.imgaug_transformations_detection = imgaug_transformations_detection
        self.imgaug_transformations_recognition = imgaug_transformations_recognition
        self.imgaug_transformations_layout = imgaug_transformations_layout
        self.failure_sample_paths = []
        # self.ready_detection_data_gen = ReadyDetectionData(rectangular_bounding_boxes, detection_data_classes_field_names, detection_data_fields_to_mask_in_loss, consider_elastic, consider_affine)
        self.log.info('Copying Datasets For Augmentation')
        self.augmented_image_paths_to_detection_data = self.__copy_detection_or_layout('detection')
        if not augment_recognition_by_detection or sum(list(self.transformations_to_probabilities_recognition.values())) > 0:
            self.augmented_image_paths_to_recognition_data = self.__copy_recognition()
        else:
            self.augmented_image_paths_to_recognition_data = {}
        self.augmented_image_paths_to_layout_data = {}  #self.__copy_detection_or_layout('layout')
        self.image_paths_for_style_transfer_model = {k: [] for k in self.style_transfer_model_names_to_probabilities.keys()}
        self.log.info('Searching for background images')
        self.natural_backgrounds = [os.path.join(par, fn) for (par, _, files) in os.walk(NATURAL_BACKGROUNDS_PATH) for fn in files if fn.lower().endswith(OCRDatasetView.IMAGES_EXT)]
        self.texture_backgrounds = [os.path.join(par, fn) for (par, _, files) in os.walk(TEXTURE_BACKGROUNDS_PATH) for fn in files if fn.lower().endswith(OCRDatasetView.IMAGES_EXT)]
        self.log.info('Start Augmentation')
        self.__augment_dataset()
        for p in self.failure_sample_paths:
            shutil.rmtree(p, ignore_errors=True)
        convert_detection_to_coco(self.augmented_dataset_folder)
        convert_layout_to_coco(self.augmented_dataset_folder)
        self.log.info('Creating DataView')
        self.augmented_dataset_view = OCRDatasetView(augmented_dataset_name,
                                                    DATASETS_VIEWS_PATH,
                                                    DATASETS_PATH,
                                                    load_view_file=False,
                                                    inference=False,
                                                    language_name=None,
                                                    language_folder=None,
                                                    running_names=[augmented_dataset_name],
                                                    detection=True,
                                                    recognition=True,
                                                    layout=True,
                                                    enable_icon_or_emoji_text=enable_icon_or_emoji_text)
        augmentation_info_file = os.path.join(self.augmented_dataset_folder, 'augmentation_info.json')
        self.log.info('Creating Augmentation Info')
        data = {
            'augmented_dataset_name': augmented_dataset_name,
            'ocr_dataset_view_name': ocr_dataset_view.dataset_view_name,
            'transformations_to_probabilities_detection': transformations_to_probabilities_detection,
            'transformations_to_probabilities_recognition': transformations_to_probabilities_recognition,
            'transformations_to_probabilities_layout': transformations_to_probabilities_layout,
            'augment_recognition_by_detection': augment_recognition_by_detection,
            'style_transfer_model_names_to_probabilities': style_transfer_model_names_to_probabilities,
            'style_transfer_model_names_to_style_transfer_type': style_transfer_model_names_to_style_transfer_type,
            'gpu_ids': gpu_ids,
            'demonstration_font_name': demonstration_font_name,
            'salt_and_pepper_prob': salt_and_pepper_prob,
            'rectangular_bounding_boxes': rectangular_bounding_boxes,
            'detection_data_classes_field_names': detection_data_classes_field_names,
            'detection_data_fields_to_mask_in_loss': detection_data_fields_to_mask_in_loss,
            'consider_elastic': consider_elastic,
            'consider_perspective': consider_perspective,
            'style_transfer_only_on_clean_background': style_transfer_only_on_clean_background,
            'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
            'imgaug_transformations_detection': str(self.imgaug_transformations_detection),
            'imgaug_transformations_recognition': str(self.imgaug_transformations_recognition),
            'imgaug_transformations_layout': str(self.imgaug_transformations_layout)
        }
        save_to_json(data, augmentation_info_file)

    def __copy_detection_or_layout(self, datatype='detection'):
        image_paths_to_detection_data_dst = {}
        if datatype == 'detection':
            image_paths_to_data = self.image_paths_to_detection_data.copy()
        else:
            image_paths_to_data = self.image_paths_to_layout_data.copy()
        if len(image_paths_to_data) > 0:
            for idx, (image_path, data) in enumerate(image_paths_to_data.items()):
                sample_folder = os.path.join(self.augmented_dataset_folder, str(idx))
                try:
                    shutil.rmtree(sample_folder, ignore_errors=True)
                    os.makedirs(sample_folder, exist_ok=True)
                    detection_folder = os.path.join(sample_folder, datatype + '_data')
                    os.makedirs(detection_folder, exist_ok=True)
                    mask_path, mask_images_path, bounding_boxes_mask_path, bounding_boxes_mask_transformed_path, bounding_boxes_image_mask_path, \
                    bounding_boxes_image_mask_transformed_path, background_path, background_images_path, data_path, elastic_indices_path = data['mask_file'], data['mask_images_file'], \
                                                                                                                  data['bounding_boxes_text_mask_file'], \
                                                                                                                  data['bounding_boxes_text_mask_transformed_file'], \
                                                                                                                  data['bounding_boxes_image_mask_file'], \
                                                                                                                  data['bounding_boxes_image_mask_transformed_file'], \
                                                                                                                  data['background_file'], data['background_images_file'], data['data_file'], data['elastic']
                    if not os.path.isfile(mask_path):
                        mask_path = os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'mask.png')
                    if not os.path.isfile(bounding_boxes_mask_path):
                        bounding_boxes_mask_path = os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_text_mask.png')
                    if not os.path.isfile(bounding_boxes_mask_transformed_path):
                        bounding_boxes_mask_transformed_path = os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_text_mask_transformed.png')
                    if not os.path.isfile(bounding_boxes_image_mask_path):
                        bounding_boxes_image_mask_path = os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_images_mask.png')
                    if not os.path.isfile(bounding_boxes_image_mask_transformed_path):
                        bounding_boxes_image_mask_transformed_path = os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_images_mask_transformed.png')
                    if elastic_indices_path is not None and not os.path.isfile(elastic_indices_path):
                        elastic_indices_path = os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'indices.npy')
                    image_path_dst = os.path.join(detection_folder, 'image.' + image_path.split('.')[-1])
                    mask_path_dst = os.path.join(detection_folder, 'mask.png')
                    mask_images_path_dst = os.path.join(detection_folder, 'mask_images.png')
                    bounding_boxes_mask_path_dst = os.path.join(detection_folder, 'bounding_boxes_words_mask.png')
                    bounding_boxes_mask_transformed_path_dst = os.path.join(detection_folder, 'bounding_boxes_words_mask_transformed.png')
                    bounding_boxes_mask_image_path_dst = os.path.join(detection_folder, 'bounding_boxes_images_mask.png')
                    bounding_boxes_mask_image_transformed_path_dst = os.path.join(detection_folder, 'bounding_boxes_images_mask_transformed.png')
                    background_path_dst = os.path.join(detection_folder, 'background.' + background_path.split('.')[-1])
                    background_images_path_dst = os.path.join(detection_folder, 'background_images.' + background_path.split('.')[-1])
                    data_path_dst = os.path.join(detection_folder, 'data.json')
                    elastic_indices_path_dst = os.path.join(detection_folder, 'indices.npy')
                    shutil.copyfile(image_path, image_path_dst)
                    shutil.copyfile(mask_path, mask_path_dst)
                    shutil.copyfile(bounding_boxes_mask_path, bounding_boxes_mask_path_dst)
                    shutil.copyfile(bounding_boxes_mask_transformed_path, bounding_boxes_mask_transformed_path_dst)
                    shutil.copyfile(bounding_boxes_image_mask_path, bounding_boxes_mask_image_path_dst)
                    shutil.copyfile(bounding_boxes_image_mask_transformed_path, bounding_boxes_mask_image_transformed_path_dst)
                    shutil.copyfile(background_path, background_path_dst)
                    shutil.copyfile(background_images_path, background_images_path_dst)
                    shutil.copyfile(mask_images_path, mask_images_path_dst)
                    data_json = json.load(open(data_path, mode='r', encoding='utf-8'))
                    if os.path.isfile(elastic_indices_path):
                        shutil.copyfile(elastic_indices_path, elastic_indices_path_dst)
                        data_json['elastic'] = elastic_indices_path_dst
                    else:
                        data_json['elastic'] = None
                    data_json['mask_file'] = mask_path_dst
                    data_json['mask_images_file'] = mask_images_path_dst
                    data_json['bounding_boxes_text_mask_file'] = bounding_boxes_mask_path_dst
                    data_json['bounding_boxes_text_mask_transformed_file'] = bounding_boxes_mask_transformed_path_dst
                    data_json['bounding_boxes_image_mask_file'] = bounding_boxes_mask_image_path_dst
                    data_json['bounding_boxes_image_mask_transformed_file'] = bounding_boxes_mask_image_transformed_path_dst
                    data_json['background_file'] = background_path_dst
                    data_json['background_images_file'] = background_images_path_dst
                    data_json['data_file'] = data_path_dst
                    save_to_json(data_json, data_path_dst)
                    image_paths_to_detection_data_dst[image_path_dst] = {'mask_file': data_json['mask_file'], 'bounding_boxes_text_mask_file': data_json['bounding_boxes_text_mask_file'],
                                                                         'bounding_boxes_text_mask_transformed_file': data_json['bounding_boxes_text_mask_transformed_file'],
                                                                         'bounding_boxes_image_mask_file': data_json['bounding_boxes_image_mask_file'],
                                                                         'bounding_boxes_image_mask_transformed_file': data_json['bounding_boxes_image_mask_transformed_file'],
                                                                         'background_file': data_json['background_file'], 'background_images_file': data_json['background_images_file'],
                                                                         'data_file': data_json['data_file'], 'image_size': data_json['image_size'],
                                                                         'background_changed': data['background_changed']}
                except Exception as e:
                    print(str(e))
                    shutil.rmtree(sample_folder, ignore_errors=True)

        return image_paths_to_detection_data_dst
    
    def __copy_recognition(self):
        image_paths_to_recognition_data_dst = {}
        if len(self.image_paths_to_recognition_data) > 0:
            recognition_folder = os.path.join(self.augmented_dataset_folder, 'recognition_data')
            shutil.rmtree(recognition_folder, ignore_errors=True)
            os.makedirs(recognition_folder, exist_ok=True)
            data_path_dst = os.path.join(recognition_folder, 'data.json')
            data_recognition = {}
            for idx, (image_path, data) in enumerate(self.image_paths_to_recognition_data.items()):
                image_path_dst = os.path.join(recognition_folder, str(idx) + '.' + image_path.split('.')[-1])
                src_mask_file_name = '.'.join(image_path.split('.')[:-1]) + '_mask.png'
                mask_path_dst = os.path.join(recognition_folder, str(idx) + '_mask.png') if os.path.isfile(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), src_mask_file_name)) else None
                shutil.copyfile(image_path, image_path_dst)
                if mask_path_dst is not None:
                    shutil.copyfile(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), src_mask_file_name), mask_path_dst)
                data_recognition[str(idx) + '.' + image_path.split('.')[-1]] = {
                    'text': data['text'],
                    'mask_file': mask_path_dst,
                    'image_size': data['image_size'],
                    'rotated': data['rotated'],
                    'direction': data['direction']
                }
                image_paths_to_recognition_data_dst[image_path_dst] = data_recognition[str(idx) + '.' + image_path.split('.')[-1]]  #data
            # write_dictionary_by_chunks(data, open(data_path_dst, mode='w', encoding='utf-8'))
            save_to_json(data_recognition, data_path_dst)
        return image_paths_to_recognition_data_dst

    def __augment_dataset(self):
        self.__style_transfer_dataset_augmentation()
        if len(self.transformations_to_probabilities_detection.keys()) > 0:
            self.imgaugTransform = self.imgaug_transformations_detection
            for image_path in tqdm(self.augmented_image_paths_to_detection_data.keys()):
                try:
                    self.__augment_sample(image_path, image_path, self.transformations_to_probabilities_detection,
                                          mask_paths=[os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'mask.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_words_mask.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_words_mask_transformed.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background.' + image_path.split('.')[-1]),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_images_mask.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_images_mask_transformed.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background_images.' + image_path.split('.')[-1]),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'mask_images.png')
                                                      ],
                                          data_file=os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'data.json'))
                except Exception as e:
                    print(str(e))
                    self.failure_sample_paths.append('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]))
        if self.augment_recognition_by_detection:
            self.__augment_recognition_by_detection()
        elif len(self.transformations_to_probabilities_recognition.keys()) > 0:
            self.imgaugTransform = self.imgaug_transformations_recognition
            for image_path in tqdm(self.augmented_image_paths_to_recognition_data.keys()):
                if self.augmented_image_paths_to_recognition_data[image_path]['mask_file'] is not None:
                    self.__augment_sample(image_path, image_path, self.transformations_to_probabilities_recognition, mask_paths=[image_path[:-4] + '_mask.png'])
                else:
                    self.__augment_sample(image_path, image_path, self.transformations_to_probabilities_recognition)
        if len(self.transformations_to_probabilities_layout.keys()) > 0:
            self.imgaugTransform = self.imgaug_transformations_layout
            for image_path in tqdm(self.augmented_image_paths_to_layout_data.keys()):
                try:
                    self.__augment_sample(image_path, image_path, self.transformations_to_probabilities_layout,
                                          mask_paths=[os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'mask.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_words_mask.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_words_mask_transformed.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background.' + image_path.split('.')[-1]),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_images_mask.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'bounding_boxes_images_mask_transformed.png'),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background_images.' + image_path.split('.')[-1]),
                                                      os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'mask_images.png')
                                                      ],
                                          data_file=os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'data.json'))
                except Exception as e:
                    print(str(e))
                    self.failure_sample_paths.append('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]))

    def __style_transfer_dataset_augmentation(self):
        # from colorthief import ColorThief
        style_transfer_model_names = []
        style_transfer_model_probs = []
        for k, v in self.style_transfer_model_names_to_probabilities.items():
            style_transfer_model_names.append(k)
            style_transfer_model_probs.append(v)
        for (augmented_image_paths_to_data, transformations_to_probabilities) in \
                [(self.augmented_image_paths_to_detection_data, self.transformations_to_probabilities_detection),
                  (self.augmented_image_paths_to_recognition_data, self.transformations_to_probabilities_recognition),
                  (self.augmented_image_paths_to_layout_data, self.transformations_to_probabilities_layout)]:
            for image_path in augmented_image_paths_to_data.keys():
                if 'style_transfer' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['style_transfer']:
                    if (not self.style_transfer_only_on_clean_background) or (not augmented_image_paths_to_data[image_path]['background_changed']):
                        if augmented_image_paths_to_data[image_path]['image_size'][0] > 7 and augmented_image_paths_to_data[image_path]['image_size'][1] > 7:
                            random_style_transfer_model_name = np.random.choice(style_transfer_model_names, p=style_transfer_model_probs)
                            self.image_paths_for_style_transfer_model[random_style_transfer_model_name].append(image_path)
                            if os.path.isfile(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background.' + image_path.split('.')[-1])):
                                self.image_paths_for_style_transfer_model[random_style_transfer_model_name].append(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background.' + image_path.split('.')[-1]))
                            if os.path.isfile(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background_images.' + image_path.split('.')[-1])):
                                self.image_paths_for_style_transfer_model[random_style_transfer_model_name].append(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'background_images.' + image_path.split('.')[-1]))
        for style_transfer_model_name in style_transfer_model_names:
            if len(self.image_paths_for_style_transfer_model[style_transfer_model_name]) > 0:
                style_transfer_model_type = self.style_transfer_model_names_to_style_transfer_type[style_transfer_model_name]
                self.log.info('{} {} will work on {} samples'.format(style_transfer_model_type, style_transfer_model_name, len(self.image_paths_for_style_transfer_model[style_transfer_model_name])))
                if style_transfer_model_type == 'cycle_gan':
                    self.cyclegan(self.image_paths_for_style_transfer_model[style_transfer_model_name], self.augmented_dataset_folder, CYCLEGAN_MODELS_PATH, style_transfer_model_name, self.gpu_ids)
                else:
                    screenshots_dir = make_tmp_folder()
                    scanned_documents_dir = make_tmp_folder()
                    shutil.copyfile(self.image_paths_for_style_transfer_model[style_transfer_model_name][0], os.path.join(scanned_documents_dir, '0.' + self.image_paths_for_style_transfer_model[style_transfer_model_name][0].split(os.sep)[-1]))
                    for image_idx, image_path in enumerate(self.image_paths_for_style_transfer_model[style_transfer_model_name]):
                        shutil.copyfile(image_path, os.path.join(screenshots_dir, str(image_idx) + '.' + image_path.split('.')[-1]))
                    munit_inference(style_transfer_model_name, ['screenshots', 'scanned_documents'], [screenshots_dir, scanned_documents_dir], {'screenshots': screenshots_dir, 'scanned_documents': scanned_documents_dir},
                                    inference_type=INFERENCE_TYPES.RANDOM_STYLE_CROSS_DOMAIN, gpu_ids=self.gpu_ids, crop=None, batch_size=1, random_recover_file_name=False)
                    for image_idx, image_path in enumerate(self.image_paths_for_style_transfer_model[style_transfer_model_name]):
                        shutil.copyfile(os.path.join(screenshots_dir, str(image_idx) + '.' + image_path.split('.')[-1]), image_path)
                    shutil.rmtree(screenshots_dir)
                    shutil.rmtree(scanned_documents_dir)
                self.log.info('{} is done'.format(style_transfer_model_name))

    def __augment_recognition_by_detection(self):
        self.augmented_image_paths_to_recognition_data = augment_recognition_by_detection_process(self.augmented_dataset_name, return_info=(len(self.transformations_to_probabilities_recognition.keys()) > 0))
        # recognition_folder = os.path.join(self.augmented_dataset_folder, 'recognition_data')
        # shutil.rmtree(recognition_folder, ignore_errors=True)
        # os.makedirs(recognition_folder, exist_ok=True)
        # json_recognition_file = os.path.join(recognition_folder, 'data.json')
        # num_samples = 0
        # file_names_to_data = {}
        # for image_path, data_json_dict in tqdm(self.augmented_image_paths_to_detection_data.items()):
        #     try:
        #         data_json = json.load(open(data_json_dict['data_file'], mode='r', encoding='utf-8'))
        #         detection_data = data_json['bounding_boxes_and_text_for_recognition']
        #         np_image = np.array(Image.open(image_path).convert('RGB'))
        #         mask_np_image = np.array(Image.open(data_json_dict['mask_file']).convert('RGB'))
        #         for (word, bounding_box) in detection_data:
        #             word_img = Image.fromarray(np_image[bounding_box['top']: bounding_box['bottom'] + 1,
        #                                        bounding_box['left']: bounding_box['right'] + 1, :]).convert('RGB')
        #             word_mask_image = Image.fromarray(mask_np_image[bounding_box['top']: bounding_box['bottom'] + 1,
        #                                        bounding_box['left']: bounding_box['right'] + 1, :]).convert('RGB')
        #             file_name = str(num_samples) + '.png'
        #             word_mask_image_path = os.path.join(recognition_folder, str(num_samples) + '_mask.png')
        #             word_img.save(os.path.join(recognition_folder, file_name), 'PNG', compress_level=0)
        #             word_mask_image.save(word_mask_image_path, 'PNG')  #, compress_level=0)
        #             word_data = {
        #                 'text': word,
        #                 'image_size': (word_img.size[1], word_img.size[0]),
        #                 'mask_file': word_mask_image_path,
        #                 'rotated': False,
        #                 'direction': data_json['direction']
        #             }
        #             file_names_to_data[file_name] = word_data
        #             self.augmented_image_paths_to_recognition_data[os.path.join(recognition_folder, file_name)] = word_data
        #             num_samples += 1
        #             if num_samples % 100000 == 0:
        #                 save_to_json(file_names_to_data, json_recognition_file)
        #     except Exception as e:
        #         self.log.warning(str(e))
        # save_to_json(file_names_to_data, json_recognition_file)

    def __augment_sample(self, src_image_path, dst_img_path, transformations_to_probabilities, mask_paths=None, data_file=None):
        perspective = False
        if src_image_path != dst_img_path:
            shutil.copyfile(src_image_path, dst_img_path)
        if 'imgaug' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['imgaug']:
            if mask_paths is not None:
                dst_img_path_include_masks = [dst_img_path] + mask_paths
            else:
                dst_img_path_include_masks = [dst_img_path]
            self.imgaug_transformation(dst_img_path_include_masks, dst_img_path_include_masks, self.imgaugTransform, data_file, self.natural_backgrounds + self.texture_backgrounds)
        if 'grayscale' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['grayscale']:
            self.grayscale(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.grayscale(mask_paths[3], mask_paths[3])
        if 'sharpen' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['sharpen']:
            self.sharpen(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.sharpen(mask_paths[3], mask_paths[3])
        if 'salt_and_pepper' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['salt_and_pepper']:
            random_matrix = self.salt_and_pepper(dst_img_path, dst_img_path, self.salt_and_pepper_prob)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.salt_and_pepper(mask_paths[3], mask_paths[3], self.salt_and_pepper_prob, random_matrix=random_matrix)
        if 'sepia' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['sepia']:
            self.sepia(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.sepia(mask_paths[3], mask_paths[3])
        if 'pencil' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['pencil']:
            self.pencil(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.pencil(mask_paths[3], mask_paths[3])
        if 'hdr' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['hdr']:
            self.hdr(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.hdr(mask_paths[3], mask_paths[3])
        if 'invert' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['invert']:
            self.invert(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.invert(mask_paths[3], mask_paths[3])
        if 'laplacian' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['laplacian']:
            self.laplacian(dst_img_path, dst_img_path)
            if mask_paths is not None and len(mask_paths) >= 6:
                self.laplacian(mask_paths[3], mask_paths[3])
        if 'elastic_distortion' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['elastic_distortion']:
            if mask_paths is not None:
                dst_img_path_include_masks = [dst_img_path] + mask_paths
            else:
                dst_img_path_include_masks = [dst_img_path]
            self.elastic_distortion(dst_img_path_include_masks, dst_img_path_include_masks, data_file)
        if 'perspective_transformation' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['perspective_transformation']:
            if mask_paths is not None:
                dst_img_path_include_masks = [dst_img_path] + mask_paths
            else:
                dst_img_path_include_masks = [dst_img_path]
            self.perspective_transformation(dst_img_path_include_masks, dst_img_path_include_masks, data_file, self.natural_backgrounds + self.texture_backgrounds)
            perspective = True
        if mask_paths is not None and 'paste_text_on_natural_background' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['paste_text_on_natural_background']:
            chosen_background = self.paste_text_on_background(dst_img_path, dst_img_path, mask_paths[0], self.natural_backgrounds)
            if mask_paths is not None and len(mask_paths) >= 7:
                self.paste_text_on_background(mask_paths[3], mask_paths[3], mask_paths[0], [chosen_background])
                self.paste_text_on_background(mask_paths[6], mask_paths[6], mask_paths[0], [chosen_background])
        elif mask_paths is not None and 'paste_text_on_texture_background' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['paste_text_on_texture_background']:
            chosen_background = self.paste_text_on_background(dst_img_path, dst_img_path, mask_paths[0], self.texture_backgrounds)
            if mask_paths is not None and len(mask_paths) >= 7:
                self.paste_text_on_background(mask_paths[3], mask_paths[3], mask_paths[0], [chosen_background])
                self.paste_text_on_background(mask_paths[6], mask_paths[6], mask_paths[0], [chosen_background])
        if not perspective and 'add_natural_background_to_image' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['add_natural_background_to_image']:
            chosen_background = self.add_background_to_image(dst_img_path, dst_img_path, self.natural_backgrounds)
            if mask_paths is not None and len(mask_paths) >= 7:
                self.add_background_to_image(mask_paths[3], mask_paths[3], [chosen_background])
                self.add_background_to_image(mask_paths[6], mask_paths[6], [chosen_background])
        elif not perspective and 'add_texture_background_to_image' in transformations_to_probabilities.keys() and np.random.rand() < transformations_to_probabilities['add_texture_background_to_image']:
            chosen_background = self.add_background_to_image(dst_img_path, dst_img_path, self.texture_backgrounds)
            if mask_paths is not None and len(mask_paths) >= 7:
                self.add_background_to_image(mask_paths[3], mask_paths[3], [chosen_background])
                self.add_background_to_image(mask_paths[6], mask_paths[6], [chosen_background])
        if data_file is not None:
            data = json.load(open(data_file, mode='r', encoding='utf-8'))
            bounding_boxes_mask_transform = cv2.imread(mask_paths[2])[:, :, 0]
            ones_y, ones_x = np.where(bounding_boxes_mask_transform == 255)
            num_ones = len(ones_y)
            num_zeros = bounding_boxes_mask_transform.shape[0] * bounding_boxes_mask_transform.shape[1] - num_ones
            data['num_pixels_each_class'] = [num_zeros, num_ones]
            save_to_json(data, data_file)
            image_path_to_text_and_bb = {dst_img_path: [(data['tags_to_text'][t], data['tags_to_bounding_boxes_text'][t]) for t in data['tags_to_text'].keys()]}
            draw_images_paths_to_bounding_boxes_and_text(image_path_to_text_and_bb, os.path.join(FONTS_PATH, self.demonstration_font_name + '.ttf'), with_initial=False,
                                                         images_paths_to_bounding_boxes_images={dst_img_path: list(data['tags_to_bounding_boxes_images'].values())})
            # self.ready_detection_data_gen.make_detection_data_ready(['{}'.format(os.sep).join(dst_img_path.split(os.sep)[:-1])])
            convert_detection_to_coco('{}'.format(os.sep).join(dst_img_path.split(os.sep)[:-1]))

    @staticmethod
    def imgaug_transformation(src_image, dst_image, imgaugTransform, data_file=None, background_paths=None):
        imgaugTransformSample = imgaugTransform.to_deterministic()
        if type(src_image) == str:
            src_image = [src_image]
        if type(dst_image) == str:
            dst_image = [dst_image]
        images = [cv2.imread(src_image[idxi]) for idxi in range(len(src_image)) if idxi in [0, 4, 7]]
        images_dst_paths = [dst_image[idxi] for idxi in range(len(dst_image)) if idxi in [0, 4, 7]]
        if data_file is not None:
            image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
            tags = []
            bounding_boxes = []
            keys = []
            for key in ['tags_to_bounding_boxes_text', 'tags_to_bounding_boxes_images',
                        'tags_to_bounding_boxes_images_not_overlapping_text', 'bounding_boxes_and_text_for_recognition']:
                if key not in ['bounding_boxes_and_text_for_recognition']:
                    for idx, (tag, bb) in enumerate(image_data[key].items()):
                        tags.append(tag)
                        bounding_boxes.append(BoundingBox(x1=bb['left'], y1=bb['top'], x2=bb['right'], y2=bb['bottom'], label='None'))
                        keys.append(key)
                else:
                    for idx, (text, bb) in enumerate(image_data[key]):
                        tags.append(None)
                        bounding_boxes.append(BoundingBox(x1=bb['left'], y1=bb['top'], x2=bb['right'], y2=bb['bottom'], label=text))
                        keys.append(key)
            bounding_boxes = BoundingBoxesOnImage(bounding_boxes, shape=images[0].shape)
        else:
            bounding_boxes = None
        if len(src_image) > 1:
            mask_paths = [i for idxi, i in enumerate(src_image) if idxi not in [0, 4, 7]]
            segmap = np.zeros((images[0].shape[0], images[0].shape[1], len(mask_paths)), dtype=np.uint8)
            for mask_idx, mask_path in enumerate(mask_paths):
                segmap[:, :, mask_idx] = np.uint8(cv2.imread(mask_path)[:, :, 0] / 255.0)
            segmap = SegmentationMapsOnImage(segmap, shape=images[0].shape)
            if bounding_boxes is not None:
                img_aug, segmap_aug, bbs_aug = imgaugTransformSample(image=images[0], segmentation_maps=segmap, bounding_boxes=bounding_boxes)
            else:
                img_aug, segmap_aug = imgaugTransformSample(image=images[0], segmentation_maps=segmap)
            for mask_idx, mask_path in enumerate(mask_paths):
                cv2.imwrite(mask_path, segmap_aug.get_arr()[:, :, mask_idx] * 255)
        else:
            if bounding_boxes is not None:
                img_aug, bbs_aug = imgaugTransformSample(image=images[0], bounding_boxes=bounding_boxes)
            else:
                img_aug = imgaugTransformSample(image=images[0])
        img_aug = [img_aug]
        for i, img in enumerate(images):
            if i > 0:
                img_aug.append(imgaugTransformSample(image=images[i]))
            cv2.imwrite(images_dst_paths[i], img_aug[i])
        if bounding_boxes is not None:
            image_data['bounding_boxes_and_text_for_recognition'] = []
            for ibb, bb_aug in enumerate(bbs_aug):
                t, k = tags[ibb], keys[ibb]
                text = bb_aug.label
                bb = {
                    'left': int(np.round(bb_aug.x1)),
                    'top': int(np.round(bb_aug.y1)),
                    'right': int(np.round(bb_aug.x2)),
                    'bottom': int(np.round(bb_aug.y2))
                }
                if t is not None:
                    image_data[k][t] = bb
                else:
                    image_data[k].append((text, bb))
            save_to_json(image_data, data_file)
            bounding_box_text_image = np.zeros(images[0].shape)
            for bb in image_data['tags_to_bounding_boxes_text'].values():
                bounding_box_text_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = 255
            Image.fromarray(bounding_box_text_image.astype(np.uint8)).convert('RGB').save(dst_image[2])  #, compress_level=0)
            bounding_box_images_image = np.zeros(images[0].shape)
            for bb in image_data['tags_to_bounding_boxes_images'].values():
                bounding_box_images_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = 255
            Image.fromarray(bounding_box_images_image.astype(np.uint8)).convert('RGB').save(dst_image[5])  #, compress_level=0)

    @staticmethod
    def grayscale(src_image, dst_image):
        image = cv2.imread(src_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dst_image, gray)

    @staticmethod
    def sharpen(src_image, dst_image):
        image = cv2.imread(src_image)
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(dst_image, img_sharpen)

    @staticmethod
    def salt_and_pepper(src_image, dst_image, prob=0.05, random_matrix=None):
        image = cv2.imread(src_image)
        if random_matrix is None:
            random_matrix = np.random.random(image.shape[:2])
        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')
        noisy_image = image.copy()
        noisy_image[random_matrix < (prob / 2)] = black
        noisy_image[random_matrix > 1 - (prob / 2)] = white
        cv2.imwrite(dst_image, noisy_image)
        return random_matrix

    @staticmethod
    def sepia(src_image, dst_image):
        image = cv2.imread(src_image)
        img_sepia = np.array(image, dtype=np.float64)  # converting to float to prevent loss
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                        [0.349, 0.686, 0.168],
                                        [0.393, 0.769, 0.189]]))  # multipying image with special sepia matrix
        img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
        cv2.imwrite(dst_image, img_sepia)

    @staticmethod
    def pencil(src_image, dst_image):
        image = cv2.imread(src_image)
        sk_gray, sk_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        cv2.imwrite(dst_image, sk_color)

    @staticmethod
    def hdr(src_image, dst_image):
        image = cv2.imread(src_image)
        hdr_image = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
        cv2.imwrite(dst_image, hdr_image)

    @staticmethod
    def invert(src_image, dst_image):
        image = cv2.imread(src_image)
        inv_image = cv2.bitwise_not(image)
        cv2.imwrite(dst_image, inv_image)

    @staticmethod
    def laplacian(src_image, dst_image):
        image = cv2.imread(src_image)
        new_image = cv2.Laplacian(image, cv2.CV_64F)
        cv2.imwrite(dst_image, new_image)

    @staticmethod
    def elastic_distortion(src_image, dst_image, data_file=None):
        if type(src_image) == str:
            src_image = [src_image]
        if type(dst_image) == str:
            dst_image = [dst_image]
        images = [cv2.imread(image) for image in src_image]
        new_images = elastic(images, alpha=1000, sigma=np.random.randint(7, 10), random_state=None, data_file=data_file)
        for idx in range(len(dst_image)):
            cv2.imwrite(dst_image[idx], new_images[idx])

    @staticmethod
    def paste_text_on_background(src_image, dst_image, mask_path, background_paths):
        image = cv2.imread(src_image)
        mask = cv2.imread(mask_path)
        mask[mask > 0] = 1
        chosen_background = np.random.choice(background_paths)
        background = cv2.imread(chosen_background)
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
        background[mask > 0] = 0
        new_image = background + mask * image
        cv2.imwrite(dst_image, new_image)
        return chosen_background

    @staticmethod
    def add_background_to_image(src_image, dst_image, background_paths):
        image = cv2.imread(src_image)
        chosen_background = np.random.choice(background_paths)
        background = cv2.imread(chosen_background)
        background = cv2.resize(background, (image.shape[1], image.shape[0]))
        new_image = cv2.addWeighted(background, 0.3, image, 0.7, 0)
        cv2.imwrite(dst_image, new_image)
        return chosen_background

    @staticmethod
    def perspective_transformation(src_image, dst_image, data_file=None, background_paths=None):
        if type(src_image) == str:
            src_image = [src_image]
        if type(dst_image) == str:
            dst_image = [dst_image]
        images = [cv2.imread(image) for image in src_image]
        width = images[0].shape[1]
        height = images[0].shape[0]
        input_p = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        output_p = np.float32([[np.random.randint(0, max(int(0.3 * width), 1)), np.random.randint(0, max(int(0.3 * height), 1))],
                               [np.random.randint(0, max(int(0.3 * width), 1)), height - 1 - np.random.randint(0, max(int(0.3 * height), 1))],
                             [width - 1 - np.random.randint(0, max(int(0.3 * width), 1)), height - 1 - np.random.randint(0, max(int(0.3 * height), 1))],
                               [width - 1 - np.random.randint(0, max(int(0.3 * width), 1)), np.random.randint(0, max(int(0.3 * height), 1))]])
        matrix = cv2.getPerspectiveTransform(input_p, output_p)
        chosen_background = np.random.choice(background_paths)
        for idx, image in enumerate(images):
            if idx not in [2, 5] or data_file is None:
                new_image = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_NEAREST)  # nearest neighbor interpolation keeps the original pixels values !!
                if (idx in [0, 4, 7]) and background_paths is not None:
                    background = cv2.imread(chosen_background)
                    background = cv2.resize(background, (image.shape[1], image.shape[0]))
                    # background[new_image > 0] = 0
                    new_image[new_image > 0] = cv2.addWeighted(background, 0.3, new_image, 0.7, 0)[new_image > 0]
                    new_image[new_image == 0] = background[new_image == 0]
                cv2.imwrite(dst_image[idx], new_image)
        if data_file is not None:
            image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
            for key in ['tags_to_bounding_boxes_text',
                        'tags_to_bounding_boxes_images',
                        'tags_to_bounding_boxes_images_not_overlapping_text',
                        'bounding_boxes_and_text_for_recognition']:
                if key not in ['bounding_boxes_and_text_for_recognition']:
                    for tag in image_data[key].keys():
                        bb = image_data[key][tag]
                        left1, top1, scale = np.matmul(matrix, np.array([bb['left'], bb['top'], 1]))
                        left1, top1 = left1 / scale, top1 / scale
                        right2, bottom2, scale = np.matmul(matrix, np.array([bb['right'], bb['bottom'], 1]))
                        right2, bottom2 = right2 / scale, bottom2 / scale
                        right3, top3, scale = np.matmul(matrix, np.array([bb['right'], bb['top'], 1]))
                        right3, top3 = right3 / scale, top3 / scale
                        left4, bottom4, scale = np.matmul(matrix, np.array([bb['left'], bb['bottom'], 1]))
                        left4, bottom4 = left4 / scale, bottom4 / scale
                        left = max(min(left1, left4, right2, right3), 0)
                        right = min(max(left1, left4, right2, right3), width - 1)
                        top = max(min(top1, top3, bottom2, bottom4), 0)
                        bottom = min(max(top1, top3, bottom2, bottom4), height - 1)
                        image_data[key][tag] = {
                            'top': int(top),
                            'left': int(left),
                            'bottom': int(bottom),
                            'right': int(right)
                        }
                else:
                    for idx in range(len(image_data[key])):
                        bb = image_data[key][idx][1]
                        left1, top1, scale = np.matmul(matrix, np.array([bb['left'], bb['top'], 1]))
                        left1, top1 = left1 / scale, top1 / scale
                        right2, bottom2, scale = np.matmul(matrix, np.array([bb['right'], bb['bottom'], 1]))
                        right2, bottom2 = right2 / scale, bottom2 / scale
                        right3, top3, scale = np.matmul(matrix, np.array([bb['right'], bb['top'], 1]))
                        right3, top3 = right3 / scale, top3 / scale
                        left4, bottom4, scale = np.matmul(matrix, np.array([bb['left'], bb['bottom'], 1]))
                        left4, bottom4 = left4 / scale, bottom4 / scale
                        left = max(min(left1, left4, right2, right3), 0)
                        right = min(max(left1, left4, right2, right3), width - 1)
                        top = max(min(top1, top3, bottom2, bottom4), 0)
                        bottom = min(max(top1, top3, bottom2, bottom4), height - 1)
                        image_data[key][idx][1] = {
                            'top': int(top),
                            'left': int(left),
                            'bottom': int(bottom),
                            'right': int(right)
                        }
            image_data['perspective'] = [matrix.tolist(), (width, height)]
            save_to_json(image_data, data_file)
            bounding_box_text_image = np.zeros(new_image.shape)
            for bb in image_data['tags_to_bounding_boxes_text'].values():
                bounding_box_text_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = 255
            Image.fromarray(bounding_box_text_image.astype(np.uint8)).convert('RGB').save(dst_image[2])  #, compress_level=0)
            bounding_box_images_image = np.zeros(new_image.shape)
            for bb in image_data['tags_to_bounding_boxes_images'].values():
                bounding_box_images_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = 255
            Image.fromarray(bounding_box_images_image.astype(np.uint8)).convert('RGB').save(dst_image[5])  #, compress_level=0)
        return matrix

    @staticmethod
    def cyclegan(image_paths, working_folder, cyclegan_checkpoints_folder, cyclegan_model_name, gpu_ids):
        os.makedirs(working_folder, exist_ok=True)
        tmp_folder = os.path.join(working_folder, 'tmp')
        os.makedirs(tmp_folder, exist_ok=True)
        testA_folder = os.path.join(tmp_folder, 'testA')
        os.makedirs(testA_folder, exist_ok=True)
        testB_folder = os.path.join(tmp_folder, 'testB')
        os.makedirs(testB_folder, exist_ok=True)
        results_folder = os.path.join(working_folder, 'results')
        os.makedirs(results_folder, exist_ok=True)
        for idx, image_path in enumerate(image_paths):
            shutil.copyfile(image_path, os.path.join(testA_folder, str(idx) + '.{}'.format(image_path.split('.')[-1])))
            shutil.copyfile(image_path, os.path.join(testB_folder, str(idx) + '.{}'.format(image_path.split('.')[-1])))
        os.system("{} Data/pytorch-CycleGAN-and-pix2pix-master/test.py --dataroot '{}' --checkpoints_dir '{}' --results_dir '{}' --gpu_ids {} --name {} --model cycle_gan --preprocess no".format(sys.executable, tmp_folder, cyclegan_checkpoints_folder, results_folder, gpu_ids, cyclegan_model_name))
        fakes_folder = os.path.join(results_folder, cyclegan_model_name, 'test_latest', 'images')
        for file_name in os.listdir(fakes_folder):
            if file_name.endswith('_fake_B.png'):
                idx = int('_'.join(file_name.split('_')[:-2]))
                image_path = image_paths[idx]
                shutil.copyfile(os.path.join(fakes_folder, file_name), image_path)
        shutil.rmtree(tmp_folder)
        shutil.rmtree(results_folder)


def elastic(images, alpha, sigma, random_state=None, data_file=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    image = images[0]
    if random_state is None:
      random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_images = []
    for idx, image in enumerate(images):
        distored_image = map_coordinates(image, indices, order=1, mode='nearest')
        distored_image = distored_image.reshape(image.shape)
        if idx in [1, 2, 3, 5, 6, 8]:
            distored_image[distored_image[:, :, 0] >= 128, :] = [255, 255, 255]
            distored_image[distored_image[:, :, 0] < 128, :] = [0, 0, 0]
        distored_images.append(distored_image)
    if data_file is not None:
        image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
        for key in ['tags_to_bounding_boxes_text', 'tags_to_bounding_boxes_images',
                    'tags_to_bounding_boxes_images_not_overlapping_text', 'bounding_boxes_and_text_for_recognition']:
            bounding_box_image = np.zeros(image.shape)
            idx_to_tag = {}
            num_bbs = 0
            if key not in ['bounding_boxes_and_text_for_recognition']:
                for idx, (tag, bb) in enumerate(image_data[key].items()):
                    bounding_box_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = idx + 1
                    idx_to_tag[idx + 1] = tag
                    num_bbs += 1
            else:
                for idx, (_, bb) in enumerate(image_data[key]):
                    bounding_box_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = idx + 1
                    num_bbs += 1
            distored_bounding_box_image = map_coordinates(bounding_box_image, indices, order=1, mode='nearest').reshape(image.shape)
            for idx in range(num_bbs):
                y, x = np.where(distored_bounding_box_image[:, :, 0] == (idx + 1))
                if len(y) > 0:
                    distored_bounding_box = {
                        'top': int(y.min()),
                        'left': int(x.min()),
                        'bottom': int(y.max()),
                        'right': int(x.max())
                    }
                    if key not in ['bounding_boxes_and_text_for_recognition']:
                        image_data[key][idx_to_tag[idx + 1]] = distored_bounding_box
                    else:
                        image_data[key][idx][1] = distored_bounding_box
        image_data['elastic'] = os.path.join('{}'.format(os.sep).join(data_file.split(os.sep)[:-1]), 'indices.npy')  #(alpha, sigma, (random_state.get_state()[0], random_state.get_state()[1].tolist(), random_state.get_state()[2], random_state.get_state()[3], random_state.get_state()[4]))
        np.save(os.path.join('{}'.format(os.sep).join(data_file.split(os.sep)[:-1]), 'indices.npy'), indices)
        save_to_json(image_data, data_file)
        bounding_box_text_image = np.zeros(image.shape)
        for bb in image_data['tags_to_bounding_boxes_text'].values():
            bounding_box_text_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = 255
        distored_images[2] = bounding_box_text_image
        bounding_box_images_image = np.zeros(image.shape)
        for bb in image_data['tags_to_bounding_boxes_images'].values():
            bounding_box_images_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :] = 255
        distored_images[5] = bounding_box_images_image
    return distored_images


def augment_recognition_by_detection_process(augmented_dataset_name, return_info=False):
    augmented_image_paths_to_recognition_data = {}
    num_samples = 0
    augmented_dataset_path = os.path.join(DATASETS_PATH, augmented_dataset_name)
    for fn in tqdm(os.listdir(augmented_dataset_path)):
        detection_folder = os.path.join(augmented_dataset_path, fn, 'detection_data')
        recognition_folder = os.path.join(augmented_dataset_path, fn, 'recognition_data')
        mask_file = os.path.join(detection_folder, 'mask.png')
        data_file = os.path.join(detection_folder, 'data.json')
        image_file = os.path.join(detection_folder, 'image.png')
        image_file = image_file if os.path.isfile(image_file) else os.path.join(detection_folder, 'image.jpg')
        if os.path.isfile(image_file) and os.path.isfile(mask_file):
            shutil.rmtree(recognition_folder, ignore_errors=True)
            os.makedirs(recognition_folder, exist_ok=True)
            json_recognition_file = os.path.join(recognition_folder, 'data.json')
            file_names_to_data = {}
            try:
                data_json = json.load(open(data_file, mode='r', encoding='utf-8'))
                detection_data = data_json['bounding_boxes_and_text_for_recognition']
                np_image = np.array(Image.open(image_file).convert('RGB'))
                mask_np_image = np.array(Image.open(mask_file).convert('RGB'))
                for (word, bounding_box) in detection_data:
                    word_img = Image.fromarray(np_image[bounding_box['top']: bounding_box['bottom'] + 1,
                                               bounding_box['left']: bounding_box['right'] + 1, :]).convert('RGB')
                    word_mask_image = Image.fromarray(mask_np_image[bounding_box['top']: bounding_box['bottom'] + 1,
                                               bounding_box['left']: bounding_box['right'] + 1, :]).convert('RGB')
                    file_name = str(num_samples) + '.png'
                    word_mask_image_path = os.path.join(recognition_folder, str(num_samples) + '_mask.png')
                    word_img.save(os.path.join(recognition_folder, file_name), 'PNG', compress_level=0)
                    word_mask_image.save(word_mask_image_path, 'PNG')  #, compress_level=0)
                    word_data = {
                        'text': word,
                        'image_size': (word_img.size[1], word_img.size[0]),
                        'mask_file': word_mask_image_path,
                        'rotated': False,
                        'direction': data_json['direction']
                    }
                    file_names_to_data[file_name] = word_data
                    if return_info:
                        augmented_image_paths_to_recognition_data[os.path.join(recognition_folder, file_name)] = word_data
                    num_samples += 1
                save_to_json(file_names_to_data, json_recognition_file)
            except Exception as e:
                print(str(e))
                continue
    return augmented_image_paths_to_recognition_data

