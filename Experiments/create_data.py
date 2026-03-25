import json
import string
import os
from Config.setting import *
from OcrPipeline.ocr_training_pipeline import OcrTrainingPipeline
import numpy as np


def running_data_creation(pipeline_name,
                          charset=string.digits + string.ascii_letters + string.punctuation,
                          charset_to_charset_normalization=None,
                          language_detection=None,
                          language_detection_threshold=0.5,
                          training_num_samples_V1=np.Inf,
                          val_num_samples_V1=np.Inf,
                          test_num_samples_V1=np.Inf,
                          training_num_samples_V2=0,
                          val_num_samples_V2=0,
                          test_num_samples_V2=0,
                          random_background_prob_options=(0.0,),
                          random_scrolling_prob_options=(0.5,),
                          all_page_augmentations=(0.0,),
                          words_augmentations=(0.0,),
                          direction="ltr",
                          links_name="offline_web_links",
                          prepare_recognition_data=False,
                          recognition_image_size=(48, 150),
                          recognition_resize_method="ar",
                          dictionary_names_to_probabilities=None,
                          rotation_3d_probability_V2=0.25,
                          max_random_padding_v2=5,
                          enable_icon_or_emoji_text=True,
                          probability_change_by_web_modifier=0.0,
                          webpage_modifier=None,
                          compress_datasets_to_jpeg=False,
                          ):
    if training_num_samples_V1 + val_num_samples_V1 + test_num_samples_V1 > 0:
        json_file = os.path.join(WEB_LINKS_PATH, links_name + '.json')
        web_links = json.load(open(json_file, mode='r', encoding='utf-8'))
        web_links = [i for v in web_links.values() for i in v]
    else:
        web_links = []
    for background_prob in random_background_prob_options:
        for scrolling_prob in random_scrolling_prob_options:
            for page_aug_prob in all_page_augmentations:
                for word_aug_prob in words_augmentations:
                    gen_config_name = pipeline_name + '_{}_{}_{}_{}'.format(background_prob, scrolling_prob,
                                                                            page_aug_prob, word_aug_prob)
                    OcrTrainingPipeline(
                        gen_config_name,
                        load_pipeline_file=False,
                        charset=charset,
                        direction=direction,
                        charset_to_charset_normalization=charset_to_charset_normalization,
                        language_detection=language_detection,
                        language_detection_threshold=language_detection_threshold,
                        dictionary_names_to_probabilities=dictionary_names_to_probabilities,

                        generate_V1=(training_num_samples_V1 + val_num_samples_V1 + test_num_samples_V1) > 0,
                        generate_V2=(training_num_samples_V2 + val_num_samples_V2 + test_num_samples_V2) > 0,
                        augmenting_data=False,
                        style_transfer_training=False,
                        detection_training=False,
                        recognition_training=False,

                        training_html_paths=tuple(web_links[: int(0.85 * len(web_links))]),
                        val_html_paths=tuple(web_links[int(0.85 * len(web_links)): int(0.95 * len(web_links))]),
                        test_html_paths=tuple(web_links[int(0.95 * len(web_links)):]),
                        training_num_samples_V1=training_num_samples_V1,
                        val_num_samples_V1=val_num_samples_V1,
                        test_num_samples_V1=test_num_samples_V1,
                        probability_random_scrolling_V1=scrolling_prob,
                        probability_change_background_V1=background_prob,
                        probability_change_text_color_all_page_V1=page_aug_prob,
                        probability_change_font_all_page_V1=page_aug_prob,
                        probability_change_font_size_all_page_V1=page_aug_prob,
                        probability_change_font_style_all_page_V1=page_aug_prob,
                        probability_change_font_weight_all_page_V1=page_aug_prob,
                        probability_change_font_variant_all_page_V1=page_aug_prob,
                        probability_change_font_stretch_all_page_V1=page_aug_prob,
                        probability_change_text_color_V1=word_aug_prob,
                        probability_change_font_V1=word_aug_prob,
                        probability_change_font_size_V1=word_aug_prob,
                        probability_change_font_style_V1=word_aug_prob,
                        probability_change_font_variant_V1=word_aug_prob,
                        probability_change_font_weight_V1=word_aug_prob,
                        probability_change_font_stretch_V1=word_aug_prob,
                        probability_replace_word_from_dictionary=0.0,
                        screen_appearance_change_thr=0.05,
                        force_direction_V1=direction,  # Could be also None!
                        probability_change_by_web_modifier=probability_change_by_web_modifier,
                        webpage_modifier=webpage_modifier,
                        enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                        window_size_V1=(1920, 1080),
                        window_size_V2=(1920, 1080),
                        training_num_samples_V2=training_num_samples_V2,
                        val_num_samples_V2=val_num_samples_V2,
                        test_num_samples_V2=test_num_samples_V2,
                        samples_by_order_V2=False,
                        rotation_3d_probability_V2=rotation_3d_probability_V2,
                        max_random_padding_v2=max_random_padding_v2,
                        compress_datasets_to_jpeg=compress_datasets_to_jpeg,
                        style_transfer_only_on_clean_background=True,
                        erase_elastic_data_from_datasets=True,
                        style_transfer_gpu_ids='0',
                        style_transfer_num_iters=100,
                        style_transfer_model_names_to_scanned_documents_dataset_names={},
                        augmentation_names_to_style_transfer_model_names_to_probabilities=None,
                        augmentation_names_to_transformations_to_probabilities_detection={},
                        augmentation_names_to_transformations_to_probabilities_recognition={},
                        augmentation_names_to_augment_recognition_by_detection={},
                        salt_and_pepper_prob=0.05,
                        prepare_recognition_data=prepare_recognition_data,
                        generate_layout_data=True,
                        recognition_image_size=recognition_image_size,
                        recognition_resize_method=recognition_resize_method,
                        dictionary_names='all' if dictionary_names_to_probabilities is None else list(
                            dictionary_names_to_probabilities.keys())
                    )
