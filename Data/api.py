import os
import shutil
import sys
import numpy as np
from Data.dataset_augmentation import DatasetAugmentation, augment_recognition_by_detection_process
from Data.dataset_view import OCRDatasetView, LightOCRDatasetView
from Data.dictionary_view import DictionaryView
from Data.fonts_view import FontsView
from Data.language import Language
from Data.SyntheticData.V1.detection_and_recognition import generate_ocr_data
from Data.SyntheticData.V2.v2 import generate_recognition
from Config.setting import *
from GeneralUtils.project_exceptions import *
from GeneralUtils.project_logs import create_log
from Data.prepare_datasets import combine_datasets_clean_failures_split_train_val
import uuid
from PIL import Image
from tqdm import tqdm
from Data.compress_datasets import prepare_recognition_data_for_faster_training, prepare_recognition_zip_for_external_training, prepare_detection_zip_for_external_training, validate_recognition_zips_with_jpg
from Data.conversion_to_coco import *
from Data.SyntheticData.make_detection_data_ready import ReadyDetectionData
from Data.visualize_datasets import DataVisualizer
from Data.compress_dataset_to_jpeg import compress_dataset
from Data.ExtrenalDatasets.external_dataset import MultiSetBagOfWordsExternalDataset
from Data.Munit.munit import munit_training, munit_inference, INFERENCE_TYPES
from GeneralUtils.utils import *
from imgaug import augmenters as iaa
from Data.create_synthetic_dictionaries import create_random_dictionary
from Data.create_corpuses import *
from Data.SyntheticData.DownloadWebPages.download_webpages import WebTemplateDownloader, WikiDownloader
from Data.create_links import create_links
from Data.prepare_distributed_execution_synthetic_data import *


def create_dictionary_view(dictionary_view_name, language_name=None, dictionary_names='all', dictionary_names_to_probabilities=None, max_word_length=None):
    dictionaryview = DictionaryView(dictionary_view_name,
                 DICTIONARIES_VIEWS_PATH,
                 DICTIONARIES_PATH,
                 False,
                 language_name,
                 LANGUAGES_PATH,
                 dictionary_names,
                 dictionary_names_to_probabilities,
                 max_word_length=max_word_length)
    return dictionaryview


def create_fonts_view(fonts_view_name, demonstration_font_name=None, language_name=None, font_names='all', font_names_to_probabilities=None):
    fontview = FontsView(fonts_view_name,
                 FONTS_VIEWS_PATH,
                 FONTS_PATH,
                 demonstration_font_name,
                 False,
                 language_name,
                 LANGUAGES_PATH,
                 font_names,
                 font_names_to_probabilities)
    return fontview


def create_language(language_name, charset=None, direction="ltr", charset_to_charset_normalization=None, language_detection=None, language_detection_threshold=0.5):
    language = Language(language_name, LANGUAGES_PATH, load_file=False, charset=charset, direction=direction, charset_to_charset_normalization=charset_to_charset_normalization,
                        language_detection=language_detection, language_detection_threshold=language_detection_threshold)
    return language


def create_dataset_view(dataset_view_name, load_view_file=False, inference=False, language_name=None, max_word_length=38,
                        running_names='all', detection=True, recognition=True, layout=True, to_save=True, enable_icon_or_emoji_text=False):
    dataview = OCRDatasetView(dataset_view_name, DATASETS_VIEWS_PATH, DATASETS_PATH, load_view_file, inference, language_name,
                 LANGUAGES_PATH, max_word_length, running_names, detection, recognition, layout, to_save, enable_icon_or_emoji_text)
    return dataview


def generate_ocrdata_websites(running_name, html_paths, font_view_name, num_samples=np.inf,
                              probability_random_scrolling=0.5,
                              probability_change_background=0.5,
                              probability_change_text_color_all_page=0.5,
                              probability_change_font_all_page=0.5,
                              probability_change_font_size_all_page=0.5,
                              probability_change_text_decoration_all_page=0.5,
                              probability_change_font_style_all_page=0.5,
                              probability_change_font_weight_all_page=0.5,
                              probability_change_font_variant_all_page=0.5,
                              probability_change_font_stretch_all_page=0.5,
                              probability_change_text_color=1.0,
                              probability_change_font=1.0,
                              probability_change_font_size=1.0,
                              probability_change_text_decoration=1.0,
                              probability_change_font_style=1.0,
                              probability_change_font_variant=1.0,
                              probability_change_font_weight=1.0,
                              probability_change_font_stretch=1.0,
                              window_size=(1920, 1080),
                              force_direction='ltr', log=None,
                              recognition_max_word_length=None,
                              rectangular_bounding_boxes=True,
                              detection_data_classes_field_names=('bounding_boxes_text',),
                              detection_data_fields_to_mask_in_loss=('bounding_boxes_images',),
                              consider_elastic=True,
                              consider_perspective=True,
                              enable_icon_or_emoji_text=False,
                              dictionary=None,
                              probability_replace_word_from_dictionary=0.0,
                              generate_layout_data=True,
                              screen_appearance_change_thr=0.05,
                              probability_change_by_web_modifier=0.0,
                              webpage_modifier=None
                              ):
    if os.path.isdir(os.path.join(DATASETS_PATH, running_name)):
        raise ProjectIsAlreadyExists()
    os.makedirs(os.path.join(DATASETS_PATH, running_name))
    if log is None:
        log = create_log(running_name, os.path.join(DATASETS_PATH, running_name))
    try:
        font = FontsView(font_view_name, FONTS_VIEWS_PATH, load_view_file=True)
        generate_ocr_data(html_paths, DATASETS_PATH, running_name, font, log, num_samples,
                          window_size=window_size,
                          vertical_iou_merging_threshold=0.5,
                          probability_random_scrolling=probability_random_scrolling,
                          probability_change_background=probability_change_background,
                          probability_change_text_color_all_page=probability_change_text_color_all_page,
                          probability_change_font_all_page=probability_change_font_all_page,
                          probability_change_font_size_all_page=probability_change_font_size_all_page,
                          probability_change_text_decoration_all_page=0.0,  # not supported  #probability_change_text_decoration_all_page,
                          probability_change_font_style_all_page=probability_change_font_style_all_page,
                          probability_change_font_weight_all_page=probability_change_font_weight_all_page,
                          probability_change_font_variant_all_page=probability_change_font_variant_all_page,
                          probability_change_font_stretch_all_page=probability_change_font_stretch_all_page,
                          probability_change_text_color=probability_change_text_color,
                          probability_change_font=0.0,  # not supported  #probability_change_font,
                          probability_change_font_size=probability_change_font_size,
                          probability_change_text_decoration=0.0,  # not supported  #probability_change_text_decoration,
                          probability_change_font_style=probability_change_font_style,
                          probability_change_font_variant=probability_change_font_variant,
                          probability_change_font_weight=probability_change_font_weight,
                          probability_change_font_stretch=probability_change_font_stretch,
                          force_direction=force_direction,
                          char_level=False,
                          chromedriver=CHROMEDRIVER_PATH,
                          rectangular_bounding_boxes=rectangular_bounding_boxes,
                          detection_data_classes_field_names=detection_data_classes_field_names,
                          detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss,
                          consider_elastic=consider_elastic,
                          consider_perspective=consider_perspective,
                          enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                          dictionary=dictionary,
                          probability_replace_word_from_dictionary=probability_replace_word_from_dictionary,
                          generate_layout_data=generate_layout_data,
                          screen_appearance_change_thr=screen_appearance_change_thr,
                          probability_change_by_web_modifier=probability_change_by_web_modifier,
                          webpage_modifier=webpage_modifier
                          )
        dataview = OCRDatasetView(running_name, DATASETS_VIEWS_PATH, DATASETS_PATH, load_view_file=False, inference=False, language_name=None, language_folder=None,
                        running_names=[running_name], detection=True, recognition=True, layout=True, max_word_length=recognition_max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
        log.info('Generating {} finished!'.format(running_name))
        return dataview
    except Exception as e:
        log.exception(str(e))
        raise e


def generate_recognition_data(running_name, dictionary_view_name, font_view_name, language_name, num_samples, samples_by_order=False, rotation_3d_probability=0.5, max_random_padding=5, log=None, window_size=(1920, 1080), max_word_length=None, enable_icon_or_emoji_text=False):
    if os.path.isdir(os.path.join(DATASETS_PATH, running_name)):
        raise ProjectIsAlreadyExists()
    os.makedirs(os.path.join(DATASETS_PATH, running_name))
    if log is None:
        log = create_log(running_name, os.path.join(DATASETS_PATH, running_name))
    try:
        dictionary = DictionaryView(dictionary_view_name, DICTIONARIES_VIEWS_PATH, load_view_file=True)
        font = FontsView(font_view_name, FONTS_VIEWS_PATH, load_view_file=True)
        language = Language(language_name, LANGUAGES_PATH, True)
        generate_recognition(DATASETS_PATH, running_name, log, dictionary, font, language=language, num_samples=num_samples, samples_by_order=samples_by_order, rotation_3d_probability=rotation_3d_probability,
                             chromedriver=CHROMEDRIVER_PATH, window_size=window_size, max_random_padding=max_random_padding)
        dataview = OCRDatasetView(running_name, DATASETS_VIEWS_PATH, DATASETS_PATH, load_view_file=False, inference=False, language_name=None, language_folder=None,
                        running_names=[running_name], detection=False, recognition=True, layout=False, max_word_length=max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
        log.info('Generating {} finished!'.format(running_name))
        return dataview
    except Exception as e:
        log.exception(str(e))
        raise e


def augment_dataset(augmented_dataset_name,
                    dataset_view_name,
                    transformations_to_probabilities_detection=None, transformations_to_probabilities_recognition=None,
                    transformations_to_probabilities_layout=None, style_transfer_model_names_to_probabilities=None,
                    style_transfer_model_names_to_style_transfer_type=None,
                    language_name=None, max_word_length=38,
                    augment_recognition_by_detection=True, gpu_ids='0', log=None, demonstration_font_name='arial', salt_and_pepper_prob=0.05,
                    rectangular_bounding_boxes=True,
                    detection_data_classes_field_names=('bounding_boxes_text',),
                    detection_data_fields_to_mask_in_loss=('bounding_boxes_images',),
                    consider_elastic=True,
                    consider_perspective=True,
                    style_transfer_only_on_clean_background=False,
                    enable_icon_or_emoji_text=False,
                    imgaug_transformations_detection=None,
                    imgaug_transformations_recognition=None,
                    imgaug_transformations_layout=None
                    ):
    if os.path.isdir(os.path.join(DATASETS_PATH, augmented_dataset_name)):
        raise ProjectIsAlreadyExists()
    os.makedirs(os.path.join(DATASETS_PATH, augmented_dataset_name), exist_ok=True)
    if log is None:
        log = create_log(augmented_dataset_name, os.path.join(DATASETS_PATH, augmented_dataset_name))
    log.info('Creating ' + augmented_dataset_name)
    if os.path.isfile(os.path.join(DATASETS_VIEWS_PATH, dataset_view_name + '.json')):
        log.info('Loading DatasetView')
        ocr_dataset_view = OCRDatasetView(dataset_view_name, DATASETS_VIEWS_PATH,
                                          DATASETS_PATH,
                                          load_view_file=True, inference=False,
                                          language_name=language_name,
                                          language_folder=LANGUAGES_PATH,
                                          max_word_length=max_word_length,
                                          detection=True,
                                          recognition=not augment_recognition_by_detection,
                                          layout=False,
                                          enable_icon_or_emoji_text=enable_icon_or_emoji_text)
    else:
        log.info('Creating DatasetView')
        ocr_dataset_view = OCRDatasetView(dataset_view_name,
                                          DATASETS_VIEWS_PATH,
                                          DATASETS_PATH,
                                          load_view_file=False, inference=False,
                                          running_names=[dataset_view_name],
                                          language_name=language_name,
                                          language_folder=LANGUAGES_PATH,
                                          max_word_length=max_word_length,
                                          detection=True,
                                          recognition=not augment_recognition_by_detection,
                                          layout=False,
                                          enable_icon_or_emoji_text=enable_icon_or_emoji_text)
    log.info('Original DataView Loaded')
    data_augmented = DatasetAugmentation(augmented_dataset_name,
                         log,
                         ocr_dataset_view,
                         transformations_to_probabilities_detection,
                         transformations_to_probabilities_recognition,
                         transformations_to_probabilities_layout,
                         augment_recognition_by_detection,
                         style_transfer_model_names_to_probabilities,
                         style_transfer_model_names_to_style_transfer_type,
                         gpu_ids,
                         demonstration_font_name,
                         salt_and_pepper_prob,
                         rectangular_bounding_boxes=rectangular_bounding_boxes,
                         detection_data_classes_field_names=detection_data_classes_field_names,
                         detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss,
                         consider_elastic=consider_elastic,
                         consider_perspective=consider_perspective,
                         style_transfer_only_on_clean_background=style_transfer_only_on_clean_background,
                         enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                         imgaug_transformations_detection=imgaug_transformations_detection,
                         imgaug_transformations_recognition=imgaug_transformations_recognition,
                         imgaug_transformations_layout=imgaug_transformations_layout
                         )
    return data_augmented.augmented_dataset_view

