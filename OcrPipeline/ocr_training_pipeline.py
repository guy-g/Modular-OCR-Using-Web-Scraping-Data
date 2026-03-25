import os
import string
from Detection.api import *
from Recognition.api import *
from Data.api import *
from Detection.api import train_detection
from Recognition.api import train_recognition
from GeneralUtils.project_logs import create_log
from OcrPipeline.ocr_inference_pipeline import OcrInferencePipeline
from Data.SyntheticData.V1.detection_and_recognition import MAX_WORDS_PER_PAGE_FOR_REPLACEMENT
from Evaluate.evaluate_pipelines import evaluate_pipeline_full_test_internal_dataset


class OcrTrainingPipeline:
    def __init__(self,
                 pipeline_name,
                 load_pipeline_file=False,
                 charset=string.digits + string.ascii_letters + string.punctuation,
                 direction="ltr",
                 charset_to_charset_normalization=None,
                 language_detection=None,
                 language_detection_threshold=0.5,
                 dictionary_names_to_probabilities=None,

                 generate_V1=True,
                 generate_V2=True,
                 detection_training=True,
                 recognition_training=True,
                 script_identification_training=True,
                 demonstration_font_name='Actor-Regular',

                 training_html_paths=tuple(),
                 val_html_paths=tuple(),
                 test_html_paths=tuple(),
                 training_num_samples_V1=np.Inf,
                 val_num_samples_V1=np.Inf,
                 test_num_samples_V1=np.Inf,
                 probability_random_scrolling_V1=0.5,
                 probability_change_background_V1=0.5,
                 probability_change_text_color_all_page_V1=0.0,
                 probability_change_font_all_page_V1=0.0,
                 probability_change_font_size_all_page_V1=0.0,
                 # probability_change_text_decoration_all_page_V1=0.5,
                 probability_change_font_style_all_page_V1=0.0,
                 probability_change_font_weight_all_page_V1=0.0,
                 probability_change_font_variant_all_page_V1=0.0,
                 probability_change_font_stretch_all_page_V1=0.0,
                 probability_change_text_color_V1=0.0,
                 probability_change_font_V1=0.0,
                 probability_change_font_size_V1=0.0,
                 # probability_change_text_decoration_V1=0.5,
                 probability_change_font_style_V1=0.0,
                 probability_change_font_variant_V1=0.0,
                 probability_change_font_weight_V1=0.0,
                 probability_change_font_stretch_V1=0.0,
                 probability_replace_word_from_dictionary=0.0,
                 screen_appearance_change_thr=0.05,
                 force_direction_V1='ltr',  # Could be also None!
                 probability_change_by_web_modifier=0.0,
                 webpage_modifier=None,
                 enable_icon_or_emoji_text=False,
                 window_size_V1=(1920, 1080),
                 window_size_V2=(1920, 1080),
                 training_num_samples_V2=5000,
                 val_num_samples_V2=500,
                 test_num_samples_V2=500,
                 samples_by_order_V2=False,
                 rotation_3d_probability_V2=0.0,
                 max_random_padding_v2=5,
                 compress_datasets_to_jpeg=True,
                 style_transfer_only_on_clean_background=True,
                 erase_elastic_data_from_datasets=True,
                 generate_layout_data=True,
                 dictionary_names='all',

                 training_already_generated_dataset_names=tuple(),
                 val_already_generated_dataset_names=tuple(),
                 test_already_generated_dataset_names=tuple(),

                 style_transfer_training=True,
                 style_transfer_gpu_ids='0',
                 style_transfer_num_iters=100,
                 style_transfer_model_names_to_scanned_documents_dataset_names=None,
                 augmentation_names_to_style_transfer_model_names_to_probabilities=None,
                 style_transfer_model_names_to_style_transfer_type=None,
                 style_transfer_model_names_to_kwargs=None,

                 augmenting_data=True,
                 augmentation_names_to_transformations_to_probabilities_detection=None,
                 augmentation_names_to_transformations_to_probabilities_recognition=None,
                 augmentation_names_to_augment_recognition_by_detection=None,
                 imgaug_transformations_detection=None,
                 imgaug_transformations_recognition=None,
                 imgaug_transformations_layout=None,
                 salt_and_pepper_prob=0.05,
                 detection_data_classes_field_names=('bounding_boxes_words_mask_transformed.png',),
                 detection_data_fields_to_mask_in_loss=('bounding_boxes_images_mask_transformed.png',),
                 mask_in_loss_first=True,
                 # detection_data_classes_field_names=('bounding_boxes_text',),
                 # detection_data_fields_to_mask_in_loss=('bounding_boxes_images',),
                 rectangular_bounding_boxes=True,
                 weight_by_class_num_pixels=True,
                 detection_batch_size=16,
                 crop=(320, 320),
                 detection_gpu_ids='0',
                 detection_num_epochs=100,
                 detection_lr=1e-4,
                 detection_depth=6,
                 binary_mask_threshold=0.5,
                 pretrained_detection_name=None,

                 prepare_recognition_data=False,
                 recognition_batch_size=256,
                 recognition_num_epochs=100,
                 recognition_lr=1e-5,
                 recognition_gpu_ids='0',
                 recognition_random_transformation=0.25,
                 recognition_max_word_length=1000,
                 recognition_encoder_type="bilstm",
                 recognition_hidden_size=512,
                 recognition_num_layers=4,
                 recognition_num_heads_self_attention=4,
                 recognition_dropout_self_attention=0.0,
                 recognition_without_rotated_data=False,
                 pretrained_recognition_name=None,
                 recognition_image_size=(48, 150),
                 recognition_resize_method="ar",
                 recognition_loss_func_name='nll',
                 recognition_decoder_type="fc",
                 recognition_positional_enconding_type="const",
                 recognition_init_pos_as_const=True,
                 num_workers_recognition=1,
                 early_stop_by_recognition=('val_loss', 'min'),
                 recognition_chunking_params=None,

                 detection_data_ready=True,
                 detection_padding_inference=0,
                 detection_binary_mask_threshold_inference=0.5,
                 whole_image_preprocessing_inference=None,
                 every_word_preprocessing_inference=None,
                 detection_batch_size_inference=16,
                 recognition_batch_size_inference=32,
                 inference_detection_post_processing=False,
                 inference_detection_post_processing_methods=('size_based_nms',),
                 inference_detection_post_processing_params=None,


                 num_workers_detection=1,
                 detection_random_transformation=0.25,
                 detection_gamma_scheduler=0.1,
                 detection_num_epochs_scheduler=120,
                 detection_model_class="Unet2D_Diffusers",
                 detection_accumulate=1,
                 train_uda=False, src_uda_folder='', lambda_uda=1,
                 uda_train_only_discriminator=0,

                 pretrained_script_identification_name=None,
                 script_identification_gpu_ids='0',
                 num_workers_script_identification=1,
                 script_identification_binary_threshold=0.5,
                 script_identification_chunking_params=None,
                 script_identification_decision_method='first',
                 script_identification_priority=None,

    ):
        style_transfer_model_names_to_scanned_documents_dataset_names = style_transfer_model_names_to_scanned_documents_dataset_names if style_transfer_model_names_to_scanned_documents_dataset_names is not None else {}
        style_transfer_model_names_to_style_transfer_type = style_transfer_model_names_to_style_transfer_type if style_transfer_model_names_to_style_transfer_type is not None else {}
        augmentation_names_to_transformations_to_probabilities_detection = augmentation_names_to_transformations_to_probabilities_detection if augmentation_names_to_transformations_to_probabilities_detection is not None else {}
        augmentation_names_to_transformations_to_probabilities_recognition = augmentation_names_to_transformations_to_probabilities_recognition if augmentation_names_to_transformations_to_probabilities_recognition is not None else {}
        augmentation_names_to_augment_recognition_by_detection = augmentation_names_to_augment_recognition_by_detection if augmentation_names_to_augment_recognition_by_detection is not None else {}
        style_transfer_model_names_to_kwargs = style_transfer_model_names_to_kwargs if style_transfer_model_names_to_kwargs is not None else {}
        pipeline_training_folder = os.path.join(OCR_PIPELINES_TRAINING_PATH, pipeline_name)
        consider_perspective = True
        consider_elastic = True
        if not load_pipeline_file:
            shutil.rmtree(pipeline_training_folder, ignore_errors=True)
            os.makedirs(pipeline_training_folder, exist_ok=True)
            log = create_log(pipeline_name, pipeline_training_folder)
        else:
            idx = 2
            while os.path.isfile(os.path.join(pipeline_training_folder, 'log_{}.txt'.format(idx))):
                idx += 1
            log = create_log(pipeline_name, pipeline_training_folder, os.path.join(pipeline_training_folder, 'log_{}.txt'.format(idx)))
        clear_cache(log)
        parameters = {
            'pipeline_name': pipeline_name,
            'charset': charset,
            'direction': direction,
            'charset_to_charset_normalization': charset_to_charset_normalization,
            'language_detection': language_detection,
            'language_detection_threshold': language_detection_threshold,
            'dictionary_names_to_probabilities': dictionary_names_to_probabilities,
            'generate_V1': generate_V1,
            'generate_V2': generate_V2,
            'demonstration_font_name': demonstration_font_name,
            'style_transfer_training': style_transfer_training,
            'detection_training': detection_training,
            'recognition_training': recognition_training,
            'script_identification_training': script_identification_training,
            'training_html_paths': training_html_paths,
            'val_html_paths': val_html_paths,
            'test_html_paths': test_html_paths,
            'training_num_samples_V1': training_num_samples_V1,
            'val_num_samples_V1': val_num_samples_V1,
            'test_num_samples_V1': test_num_samples_V1,
            'probability_random_scrolling_V1': probability_random_scrolling_V1,
            'probability_change_background_V1': probability_change_background_V1,
            'probability_change_text_color_all_page_V1': probability_change_text_color_all_page_V1,
            'probability_change_font_all_page_V1': probability_change_font_all_page_V1,
            'probability_change_font_size_all_page_V1': probability_change_font_size_all_page_V1,
            # 'probability_change_text_decoration_all_page_V1': probability_change_text_decoration_all_page_V1,
            'probability_change_font_style_all_page_V1': probability_change_font_style_all_page_V1,
            'probability_change_font_weight_all_page_V1': probability_change_font_weight_all_page_V1,
            'probability_change_font_variant_all_page_V1': probability_change_font_variant_all_page_V1,
            'probability_change_font_stretch_all_page_V1': probability_change_font_stretch_all_page_V1,
            'probability_change_text_color_V1': probability_change_text_color_V1,
            'probability_change_font_V1': probability_change_font_V1,
            'probability_change_font_size_V1': probability_change_font_size_V1,
            # 'probability_change_text_decoration_V1': probability_change_text_decoration_V1,
            'probability_change_font_style_V1': probability_change_font_style_V1,
            'probability_change_font_variant_V1': probability_change_font_variant_V1,
            'probability_change_font_weight_V1': probability_change_font_weight_V1,
            'probability_change_font_stretch_V1': probability_change_font_stretch_V1,
            'probability_replace_word_from_dictionary': probability_replace_word_from_dictionary,
            'screen_appearance_change_thr': screen_appearance_change_thr,
            'force_direction_V1': force_direction_V1,
            'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
            'window_size_V1': window_size_V1,
            'window_size_V2': window_size_V2,
            'training_num_samples_V2': training_num_samples_V2,
            'val_num_samples_V2': val_num_samples_V2,
            'test_num_samples_V2': test_num_samples_V2,
            'samples_by_order_V2': samples_by_order_V2,
            'rotation_3d_probability_V2': rotation_3d_probability_V2,
            'max_random_padding_v2': max_random_padding_v2,
            'compress_datasets_to_jpeg': compress_datasets_to_jpeg,
            'style_transfer_only_on_clean_background': style_transfer_only_on_clean_background,
            'erase_elastic_data_from_datasets': erase_elastic_data_from_datasets,
            'training_already_generated_dataset_names': training_already_generated_dataset_names,
            'val_already_generated_dataset_names': val_already_generated_dataset_names,
            'test_already_generated_dataset_names': test_already_generated_dataset_names,
            'style_transfer_gpu_ids': style_transfer_gpu_ids,
            'style_transfer_num_iters': style_transfer_num_iters,
            'style_transfer_model_names_to_scanned_documents_dataset_names': style_transfer_model_names_to_scanned_documents_dataset_names,
            'augmentation_names_to_style_transfer_model_names_to_probabilities': augmentation_names_to_style_transfer_model_names_to_probabilities,
            'style_transfer_model_names_to_style_transfer_type': style_transfer_model_names_to_style_transfer_type,
            'style_transfer_model_names_to_kwargs': style_transfer_model_names_to_kwargs,
            'augmenting_data': augmenting_data,
            'augmentation_names_to_transformations_to_probabilities_detection': augmentation_names_to_transformations_to_probabilities_detection,
            'augmentation_names_to_transformations_to_probabilities_recognition': augmentation_names_to_transformations_to_probabilities_recognition,
            'augmentation_names_to_augment_recognition_by_detection': augmentation_names_to_augment_recognition_by_detection,
            'salt_and_pepper_prob': salt_and_pepper_prob,
            'imgaug_transformations_detection': str(imgaug_transformations_detection),
            'imgaug_transformations_recognition': str(imgaug_transformations_recognition),
            'imgaug_transformations_layout': str(imgaug_transformations_layout),
            'detection_data_classes_field_names': detection_data_classes_field_names,
            'detection_data_fields_to_mask_in_loss': detection_data_fields_to_mask_in_loss,
            'mask_in_loss_first': mask_in_loss_first,
            'rectangular_bounding_boxes': rectangular_bounding_boxes,
            'weight_by_class_num_pixels': weight_by_class_num_pixels,
            'detection_batch_size': detection_batch_size,
            'crop': crop,
            'detection_data_ready': detection_data_ready,
            'detection_gpu_ids': detection_gpu_ids,
            'num_workers_detection': num_workers_detection,
            'detection_num_epochs': detection_num_epochs,
            'detection_lr': detection_lr,
            'detection_depth': detection_depth,
            'detection_model_class': detection_model_class,
            'binary_mask_threshold': binary_mask_threshold,
            'pretrained_detection_name': pretrained_detection_name,
            'prepare_recognition_data': prepare_recognition_data,
            'recognition_batch_size': recognition_batch_size,
            'recognition_num_epochs': recognition_num_epochs,
            'recognition_lr': recognition_lr,
            'recognition_gpu_ids': recognition_gpu_ids,
            'recognition_random_transformation': recognition_random_transformation,
            'detection_random_transformation': detection_random_transformation,
            'recognition_max_word_length': recognition_max_word_length,
            'recognition_encoder_type': recognition_encoder_type,
            'recognition_hidden_size': recognition_hidden_size,
            'recognition_num_layers': recognition_num_layers,
            'recognition_num_heads_self_attention': recognition_num_heads_self_attention,
            'recognition_dropout_self_attention': recognition_dropout_self_attention,
            'recognition_without_rotated_data': recognition_without_rotated_data,
            'pretrained_recognition_name': pretrained_recognition_name,
            'recognition_image_size': recognition_image_size,
            'recognition_resize_method': recognition_resize_method,
            'recognition_loss_func_name': recognition_loss_func_name,
            'recognition_decoder_type': recognition_decoder_type,
            'recognition_positional_enconding_type': recognition_positional_enconding_type,
            'recognition_init_pos_as_const': recognition_init_pos_as_const,
            'num_workers_recognition': num_workers_recognition,
            'early_stop_by_recognition': early_stop_by_recognition,
            'recognition_chunking_params': recognition_chunking_params,
            'detection_padding_inference': detection_padding_inference,
            'detection_binary_mask_threshold_inference': detection_binary_mask_threshold_inference,
            'inference_detection_post_processing': inference_detection_post_processing,
            'inference_detection_post_processing_methods': inference_detection_post_processing_methods,
            'inference_detection_post_processing_params': inference_detection_post_processing_params,
            'whole_image_preprocessing_inference': whole_image_preprocessing_inference,
            'every_word_preprocessing_inference': every_word_preprocessing_inference,
            'detection_batch_size_inference': detection_batch_size_inference,
            'recognition_batch_size_inference': recognition_batch_size_inference,
            'pretrained_script_identification_name': pretrained_script_identification_name,
            'script_identification_gpu_ids': script_identification_gpu_ids,
            'num_workers_script_identification': num_workers_script_identification,
            'script_identification_binary_threshold': script_identification_binary_threshold,
            'script_identification_chunking_params': script_identification_chunking_params,
            'generate_layout_data': generate_layout_data
        }
        json_object = json.dumps(parameters, indent=4)
        with open(os.path.join(pipeline_training_folder, 'running_info.json'), mode='w', encoding='utf-8') as data_file:
            data_file.write(json_object)
        try:
            self.generate_V1 = generate_V1
            self.generate_V2 = generate_V2
            self.style_transfer_training = style_transfer_training
            self.detection_training = detection_training
            self.recognition_training = recognition_training
            self.training_already_generated_dataset_names = list(training_already_generated_dataset_names)
            self.val_already_generated_dataset_names = list(val_already_generated_dataset_names)
            self.test_already_generated_dataset_names = list(test_already_generated_dataset_names)
            self.augmenting_data = augmenting_data
            self.compress_datasets_to_jpeg = compress_datasets_to_jpeg
            log.info('Create Language')
            self.language = create_language(pipeline_name, charset=charset, direction=direction, charset_to_charset_normalization=charset_to_charset_normalization, language_detection=language_detection, language_detection_threshold=language_detection_threshold)
            if (self.generate_V1 or self.generate_V2) and (dictionary_names_to_probabilities is None or pipeline_name in dictionary_names_to_probabilities.keys()):
                log.info('Create random default dictionary from charset')
                charset_list = list(self.language.accepting_charset)
                random_dict = []
                for _ in tqdm(range(training_num_samples_V2 + val_num_samples_V2 + test_num_samples_V2 + MAX_WORDS_PER_PAGE_FOR_REPLACEMENT)):
                    word = ''.join(np.random.choice(charset_list, np.random.randint(1, recognition_max_word_length + 1)))
                    random_dict.append(word)
                with open(os.path.join(DICTIONARIES_PATH, pipeline_name + '_random_default_charset.txt'), mode='a', encoding='utf-8') as f:
                    for word in tqdm(random_dict):
                        f.write(word + '\n')
            if self.generate_V1 or self.generate_V2:
                log.info('Create Font')
                self.font = create_fonts_view(pipeline_name, demonstration_font_name=demonstration_font_name, language_name=pipeline_name)
            if self.generate_V2:
                log.info('Create Dictionary')
                self.dictionary = create_dictionary_view(pipeline_name, language_name=pipeline_name, dictionary_names=dictionary_names, dictionary_names_to_probabilities=dictionary_names_to_probabilities, max_word_length=recognition_max_word_length)
            else:
                self.dictionary = None
            if self.generate_V1:
                if training_num_samples_V1 > 0 and len(training_html_paths) > 0:
                    generate_ocrdata_websites(pipeline_name + '_train_V1', training_html_paths, pipeline_name, training_num_samples_V1,
                                              probability_random_scrolling=probability_random_scrolling_V1,
                                              probability_change_background=probability_change_background_V1,
                                              probability_change_text_color_all_page=probability_change_text_color_all_page_V1,
                                              probability_change_font_all_page=probability_change_font_all_page_V1,
                                              probability_change_font_size_all_page=probability_change_font_size_all_page_V1,
                                              # probability_change_text_decoration_all_page=probability_change_text_decoration_all_page_V1,
                                              probability_change_font_style_all_page=probability_change_font_style_all_page_V1,
                                              probability_change_font_weight_all_page=probability_change_font_weight_all_page_V1,
                                              probability_change_font_variant_all_page=probability_change_font_variant_all_page_V1,
                                              probability_change_font_stretch_all_page=probability_change_font_stretch_all_page_V1,
                                              probability_change_text_color=probability_change_text_color_V1,
                                              probability_change_font=probability_change_font_V1,
                                              probability_change_font_size=probability_change_font_size_V1,
                                              # probability_change_text_decoration=probability_change_text_decoration_V1,
                                              probability_change_font_style=probability_change_font_style_V1,
                                              probability_change_font_variant=probability_change_font_variant_V1,
                                              probability_change_font_weight=probability_change_font_weight_V1,
                                              probability_change_font_stretch=probability_change_font_stretch_V1,
                                              window_size=window_size_V1, force_direction=force_direction_V1, log=log,
                                              recognition_max_word_length=recognition_max_word_length,
                                              rectangular_bounding_boxes=rectangular_bounding_boxes,
                                              detection_data_classes_field_names=detection_data_classes_field_names,
                                              detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss,
                                              consider_elastic=consider_elastic,
                                              consider_perspective=consider_perspective,
                                              enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                                              dictionary=self.dictionary,
                                              probability_replace_word_from_dictionary=probability_replace_word_from_dictionary,
                                              generate_layout_data=generate_layout_data,
                                              screen_appearance_change_thr=screen_appearance_change_thr,
                                              probability_change_by_web_modifier=probability_change_by_web_modifier,
                                              webpage_modifier=webpage_modifier
                    )
                    self.training_already_generated_dataset_names.append(pipeline_name + '_train_V1')
                if val_num_samples_V1 > 0 and len(val_html_paths) > 0:
                    generate_ocrdata_websites(pipeline_name + '_val_V1', val_html_paths, pipeline_name, val_num_samples_V1,
                                              probability_random_scrolling=probability_random_scrolling_V1,
                                              probability_change_background=probability_change_background_V1,
                                              probability_change_text_color_all_page=probability_change_text_color_all_page_V1,
                                              probability_change_font_all_page=probability_change_font_all_page_V1,
                                              probability_change_font_size_all_page=probability_change_font_size_all_page_V1,
                                              # probability_change_text_decoration_all_page=probability_change_text_decoration_all_page_V1,
                                              probability_change_font_style_all_page=probability_change_font_style_all_page_V1,
                                              probability_change_font_weight_all_page=probability_change_font_weight_all_page_V1,
                                              probability_change_font_variant_all_page=probability_change_font_variant_all_page_V1,
                                              probability_change_font_stretch_all_page=probability_change_font_stretch_all_page_V1,
                                              probability_change_text_color=probability_change_text_color_V1,
                                              probability_change_font=probability_change_font_V1,
                                              probability_change_font_size=probability_change_font_size_V1,
                                              # probability_change_text_decoration=probability_change_text_decoration_V1,
                                              probability_change_font_style=probability_change_font_style_V1,
                                              probability_change_font_variant=probability_change_font_variant_V1,
                                              probability_change_font_weight=probability_change_font_weight_V1,
                                              probability_change_font_stretch=probability_change_font_stretch_V1,
                                              window_size=window_size_V1, force_direction=force_direction_V1, log=log,
                                              recognition_max_word_length=recognition_max_word_length,
                                              rectangular_bounding_boxes=rectangular_bounding_boxes,
                                              detection_data_classes_field_names=detection_data_classes_field_names,
                                              detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss,
                                              consider_elastic=consider_elastic,
                                              consider_perspective=consider_perspective,
                                              enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                                              dictionary=self.dictionary,
                                              probability_replace_word_from_dictionary=probability_replace_word_from_dictionary,
                                              generate_layout_data=generate_layout_data,
                                              screen_appearance_change_thr=screen_appearance_change_thr,
                                              probability_change_by_web_modifier=probability_change_by_web_modifier,
                                              webpage_modifier=webpage_modifier
                                              )
                    self.val_already_generated_dataset_names.append(pipeline_name + '_val_V1')
                if test_num_samples_V1 > 0 and len(test_html_paths) > 0:
                    generate_ocrdata_websites(pipeline_name + '_test_V1', test_html_paths, pipeline_name, test_num_samples_V1,
                                              probability_random_scrolling=probability_random_scrolling_V1,
                                              probability_change_background=probability_change_background_V1,
                                              probability_change_text_color_all_page=probability_change_text_color_all_page_V1,
                                              probability_change_font_all_page=probability_change_font_all_page_V1,
                                              probability_change_font_size_all_page=probability_change_font_size_all_page_V1,
                                              # probability_change_text_decoration_all_page=probability_change_text_decoration_all_page_V1,
                                              probability_change_font_style_all_page=probability_change_font_style_all_page_V1,
                                              probability_change_font_weight_all_page=probability_change_font_weight_all_page_V1,
                                              probability_change_font_variant_all_page=probability_change_font_variant_all_page_V1,
                                              probability_change_font_stretch_all_page=probability_change_font_stretch_all_page_V1,
                                              probability_change_text_color=probability_change_text_color_V1,
                                              probability_change_font=probability_change_font_V1,
                                              probability_change_font_size=probability_change_font_size_V1,
                                              # probability_change_text_decoration=probability_change_text_decoration_V1,
                                              probability_change_font_style=probability_change_font_style_V1,
                                              probability_change_font_variant=probability_change_font_variant_V1,
                                              probability_change_font_weight=probability_change_font_weight_V1,
                                              probability_change_font_stretch=probability_change_font_stretch_V1,
                                              window_size=window_size_V1, force_direction=force_direction_V1, log=log,
                                              recognition_max_word_length=recognition_max_word_length,
                                              rectangular_bounding_boxes=rectangular_bounding_boxes,
                                              detection_data_classes_field_names=detection_data_classes_field_names,
                                              detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss,
                                              consider_elastic=consider_elastic,
                                              consider_perspective=consider_perspective,
                                              enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                                              dictionary=self.dictionary,
                                              probability_replace_word_from_dictionary=probability_replace_word_from_dictionary,
                                              generate_layout_data=generate_layout_data,
                                              screen_appearance_change_thr=screen_appearance_change_thr,
                                              probability_change_by_web_modifier=probability_change_by_web_modifier,
                                              webpage_modifier=webpage_modifier
                                              )
                    self.test_already_generated_dataset_names.append(pipeline_name + '_test_V1')

            if self.generate_V2:
                if training_num_samples_V2 > 0:
                    generate_recognition_data(pipeline_name + '_train_V2', pipeline_name, pipeline_name, pipeline_name, training_num_samples_V2, samples_by_order=samples_by_order_V2, rotation_3d_probability=rotation_3d_probability_V2,
                                              max_random_padding=max_random_padding_v2, log=log,
                                              window_size=window_size_V2, max_word_length=recognition_max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
                    self.training_already_generated_dataset_names.append(pipeline_name + '_train_V2')
                if val_num_samples_V2 > 0:
                    generate_recognition_data(pipeline_name + '_val_V2', pipeline_name, pipeline_name, pipeline_name, val_num_samples_V2, samples_by_order=samples_by_order_V2, rotation_3d_probability=rotation_3d_probability_V2,
                                              max_random_padding=max_random_padding_v2, log=log,
                                              window_size=window_size_V2, max_word_length=recognition_max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
                    self.val_already_generated_dataset_names.append(pipeline_name + '_val_V2')
                if test_num_samples_V2 > 0:
                    generate_recognition_data(pipeline_name + '_test_V2', pipeline_name, pipeline_name, pipeline_name, test_num_samples_V2, samples_by_order=samples_by_order_V2, rotation_3d_probability=rotation_3d_probability_V2,
                                              max_random_padding=max_random_padding_v2, log=log,
                                              window_size=window_size_V2, max_word_length=recognition_max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
                    self.test_already_generated_dataset_names.append(pipeline_name + '_test_V2')

            if self.style_transfer_training:
                for style_transfer_model_name, scanned_documents_dataset_names in style_transfer_model_names_to_scanned_documents_dataset_names.items():
                    log.info('Start Training the {} Model : {}'.format(style_transfer_model_names_to_style_transfer_type[style_transfer_model_name], style_transfer_model_name))
                    style_transfer_training_func = train_cyclegan if style_transfer_model_names_to_style_transfer_type[style_transfer_model_name] == 'cycle_gan' else train_munit
                    style_transfer_training_func(style_transfer_model_name,
                                   self.training_already_generated_dataset_names,
                                   scanned_documents_dataset_names,
                                   style_transfer_gpu_ids,
                                   num_iters=style_transfer_num_iters,
                                   log=log,
                                   **style_transfer_model_names_to_kwargs[style_transfer_model_name])

            if self.augmenting_data:
                num_style_transfer_model_names = float(len(style_transfer_model_names_to_scanned_documents_dataset_names.keys()))
                if augmentation_names_to_style_transfer_model_names_to_probabilities is None:
                    augmentation_names_to_style_transfer_model_names_to_probabilities = {}
                    for k in augmentation_names_to_transformations_to_probabilities_detection.keys():
                        augmentation_names_to_style_transfer_model_names_to_probabilities[k] = {}
                        if self.style_transfer_training and num_style_transfer_model_names > 0:
                            for style_transfer_model_name in style_transfer_model_names_to_scanned_documents_dataset_names.keys():
                                augmentation_names_to_style_transfer_model_names_to_probabilities[k][style_transfer_model_name] = 1.0 / num_style_transfer_model_names
                elif self.style_transfer_training and num_style_transfer_model_names > 0:
                    for k in augmentation_names_to_style_transfer_model_names_to_probabilities.keys():
                        sum_values = sum(augmentation_names_to_style_transfer_model_names_to_probabilities[k].values())
                        model_names_that_dont_include = [model_name for model_name in style_transfer_model_names_to_scanned_documents_dataset_names.keys() if model_name not in augmentation_names_to_style_transfer_model_names_to_probabilities[k].keys()]
                        for model_name in model_names_that_dont_include:
                            augmentation_names_to_style_transfer_model_names_to_probabilities[k][model_name] = (1.0 - sum_values) / float(len(model_names_that_dont_include))
                for destination_type in ['train', 'val', 'test']:
                    if destination_type == 'train':
                        dataset_names = self.training_already_generated_dataset_names.copy()
                    elif destination_type == 'val':
                        dataset_names = self.val_already_generated_dataset_names.copy()
                    else:
                        dataset_names = self.test_already_generated_dataset_names.copy()
                    log.info('Augment Our Datasets')
                    for dataset_name in dataset_names:
                        for k in augmentation_names_to_style_transfer_model_names_to_probabilities.keys():
                            augmented_dataset_name = dataset_name + '_' + k + '_augmented'
                            augment_dataset(
                                augmented_dataset_name,
                                dataset_name,
                                transformations_to_probabilities_detection=augmentation_names_to_transformations_to_probabilities_detection[k],
                                transformations_to_probabilities_recognition=augmentation_names_to_transformations_to_probabilities_recognition[k],
                                transformations_to_probabilities_layout={},
                                style_transfer_model_names_to_probabilities=augmentation_names_to_style_transfer_model_names_to_probabilities[k],
                                style_transfer_model_names_to_style_transfer_type=style_transfer_model_names_to_style_transfer_type,
                                language_name=pipeline_name,
                                max_word_length=recognition_max_word_length,
                                augment_recognition_by_detection=augmentation_names_to_augment_recognition_by_detection[k],
                                gpu_ids=style_transfer_gpu_ids,
                                log=log,
                                demonstration_font_name='arial',
                                salt_and_pepper_prob=salt_and_pepper_prob,
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
                            if destination_type == 'train':
                                self.training_already_generated_dataset_names.append(augmented_dataset_name)
                            elif destination_type == 'val':
                                self.val_already_generated_dataset_names.append(augmented_dataset_name)
                            else:
                                self.test_already_generated_dataset_names.append(augmented_dataset_name)
            if self.compress_datasets_to_jpeg:
                log.info('Compress Datasets')
                for destination_type in ['train', 'val', 'test']:
                    if destination_type == 'train':
                        dataset_names = self.training_already_generated_dataset_names.copy()
                    elif destination_type == 'val':
                        dataset_names = self.val_already_generated_dataset_names.copy()
                    else:
                        dataset_names = self.test_already_generated_dataset_names.copy()
                    compress_dataset(dataset_names,
                                     language_name=pipeline_name,
                                     max_word_length=recognition_max_word_length,
                                     erase_elastic_data=erase_elastic_data_from_datasets,
                                     enable_icon_or_emoji_text=enable_icon_or_emoji_text)
            if prepare_recognition_data:
                log.info('Prepare Recognition Datasets')
                for destination_type in ['train', 'val', 'test']:
                    if destination_type == 'train':
                        dataset_names = self.training_already_generated_dataset_names.copy()
                    elif destination_type == 'val':
                        dataset_names = self.val_already_generated_dataset_names.copy()
                    else:
                        dataset_names = self.test_already_generated_dataset_names.copy()
                    for dataset_name in dataset_names:
                        prepare_recognition_data_for_faster_training(
                            os.path.join(DATASETS_PATH, dataset_name),
                            recognition_image_size,
                            recognition_resize_method
                        )
            if self.detection_training:
                self.detection_training_data = train_detection(pipeline_name, self.training_already_generated_dataset_names, self.val_already_generated_dataset_names,
                                detection_data_classes_field_names=detection_data_classes_field_names,
                                detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss,
                                rectangular_bounding_boxes=rectangular_bounding_boxes,
                                weight_by_class_num_pixels=weight_by_class_num_pixels,
                                classes_colors=((0, 0, 0), (1, 1, 1)),
                                num_epochs=detection_num_epochs, batch_size=detection_batch_size, lr=detection_lr,
                                crop=crop, depth=detection_depth, gpu_ids=detection_gpu_ids,
                                pretrained_detection_name=pretrained_detection_name,
                                binary_mask_threshold=binary_mask_threshold,
                                log=log,
                                consider_perspective=consider_perspective,
                                consider_elastic=consider_elastic,
                                detection_data_ready=detection_data_ready,
                                mask_in_loss_first=mask_in_loss_first, enable_icon_or_emoji_text=enable_icon_or_emoji_text, num_workers=num_workers_detection,
                                random_transformation=detection_random_transformation, gamma_scheduler=detection_gamma_scheduler, num_epochs_scheduler=detection_num_epochs_scheduler, model_class=detection_model_class,
                                                               train_uda=train_uda,
                                                               src_uda_folder=src_uda_folder, accumulate=detection_accumulate, lambda_uda=lambda_uda, uda_train_only_discriminator=uda_train_only_discriminator
                                                               )

            if self.recognition_training:
                self.recognition_training_data = train_recognition(pipeline_name, self.training_already_generated_dataset_names, self.val_already_generated_dataset_names,
                                  pipeline_name, random_transformation=recognition_random_transformation,
                                  max_word_length=recognition_max_word_length,
                                  num_epochs=recognition_num_epochs, batch_size=recognition_batch_size,
                                  lr=recognition_lr, loss_func_name=recognition_loss_func_name, gpu_ids=recognition_gpu_ids,
                                  encoder_type=recognition_encoder_type,
                                  hidden_size=recognition_hidden_size, num_layers=recognition_num_layers,
                                  num_heads_self_attention=recognition_num_heads_self_attention,
                                  dropout_self_attention=recognition_dropout_self_attention,
                                  without_rotated_data=recognition_without_rotated_data,
                                  pretrained_recognition_name=pretrained_recognition_name,
                                  log=log, image_size=recognition_image_size, recognition_resize_method=recognition_resize_method, already_normalized=True,
                                  decoder_type=recognition_decoder_type, positional_enconding_type=recognition_positional_enconding_type, init_pos_as_const=recognition_init_pos_as_const,
                                  enable_icon_or_emoji_text=enable_icon_or_emoji_text, num_workers=num_workers_recognition, early_stop_by=early_stop_by_recognition)
            log.info('Creating OCR Inference Pipeline')

            self.inference_pipeline = OcrInferencePipeline(pipeline_name,
                                                             os.path.join(FONTS_PATH, '{}.ttf'.format(demonstration_font_name)),
                                                             pipeline_name if detection_training else pretrained_detection_name,
                                                             [pipeline_name if recognition_training else pretrained_recognition_name],
                                                             pretrained_script_identification_name,
                                                             False,
                                                             detection_batch_size_inference,
                                                             crop,
                                                             detection_gpu_ids,
                                                             num_workers_detection,
                                                             detection_padding_inference,
                                                             detection_binary_mask_threshold_inference,
                                                             inference_detection_post_processing,
                                                             inference_detection_post_processing_methods,
                                                             inference_detection_post_processing_params,
                                                             recognition_batch_size_inference,
                                                             [recognition_gpu_ids],
                                                             num_workers_recognition,
                                                             None,
                                                             whole_image_preprocessing_inference,
                                                             every_word_preprocessing_inference,
                                                             script_identification_gpu_ids,
                                                             num_workers_script_identification,
                                                             script_identification_binary_threshold,
                                                             script_identification_decision_method,
                                                             script_identification_priority,
                                                             recognition_chunking_params,
                                                             script_identification_chunking_params,
                                                             use_onnx=False
                                                           )

            self.tests_scores = {}
            if len(self.test_already_generated_dataset_names) > 0 and ((pipeline_name if detection_training else pretrained_detection_name) is not None)\
                        or ((pipeline_name if recognition_training else pretrained_recognition_name) is not None):
                log.info('Start testing...')
                for test_name in self.test_already_generated_dataset_names:
                    log.info('Test on ' + test_name)
                    result_file = os.path.join(pipeline_training_folder, test_name + '_evaluation.json')
                    self.tests_scores[test_name] = evaluate_pipeline_full_test_internal_dataset(pipeline_name, test_name, self.language, save_to=result_file, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
            log.info(str(self.tests_scores))
            log.info('OCR Pipeline training finished!')

        except Exception as e:
            log.exception(str(e))
            raise e

