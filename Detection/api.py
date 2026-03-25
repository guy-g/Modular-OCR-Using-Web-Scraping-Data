from Config.setting import *
from Detection.detection_inference import *
from Detection.detection_train import train
from GeneralUtils.project_exceptions import *
from GeneralUtils.project_logs import create_log
import os
import uuid
from Detection.post_processing import DetectionPostProcessing


def train_detection(running_name, train_datasets_names, val_datasets_names,
                    detection_data_classes_field_names=('bounding_boxes_text',),
                    detection_data_fields_to_mask_in_loss=('bounding_boxes_images',),
                    rectangular_bounding_boxes=True,
                    weight_by_class_num_pixels=True,
                    classes_colors=((0, 0, 0), (1, 1, 1)),  # include background
                    num_epochs=100, batch_size=16, lr=1e-5, crop=(320, 320), depth=6, gpu_ids='0',
                    binary_mask_threshold=0.5,
                    pretrained_detection_name=None, log=None, consider_perspective=True, consider_elastic=True,
                    detection_data_ready=True,
                    mask_in_loss_first=True, enable_icon_or_emoji_text=False, num_workers=1, random_transformation=0.25,
                    gamma_scheduler=0.1, num_epochs_scheduler=120, model_class="Unet2D_Diffusers", train_uda=False, src_uda_folder='',
                    accumulate=1, lambda_uda=1, uda_train_only_discriminator=0):
    os.makedirs(os.path.join(DETECTION_MODELS_PATH, running_name), exist_ok=True)
    if log is None:
        log = create_log(running_name, os.path.join(DETECTION_MODELS_PATH, running_name))
    try:
        running_data = train(running_name,
                             train_datasets_names,
                             val_datasets_names,
                             DATASETS_VIEWS_PATH,
                             DATASETS_PATH,
                             DETECTION_MODELS_PATH,
                             log,
                             detection_data_classes_field_names,
                             detection_data_fields_to_mask_in_loss,
                             rectangular_bounding_boxes,
                             weight_by_class_num_pixels,
                             classes_colors,
                             num_epochs,
                             batch_size,
                             lr,
                             crop,
                             depth,
                             gpu_ids,
                             binary_mask_threshold,
                             pretrained_detection_name,
                             consider_perspective=consider_perspective,
                             consider_elastic=consider_elastic,
                             detection_data_ready=detection_data_ready,
                             mask_in_loss_first=mask_in_loss_first,
                             enable_icon_or_emoji_text=enable_icon_or_emoji_text,
                             num_workers=num_workers,
                             random_transformation=random_transformation,
                             gamma_scheduler=gamma_scheduler,
                             num_epochs_scheduler=num_epochs_scheduler,
                             accumulate=accumulate,
                             model_class=model_class,
                             train_uda=train_uda,
                             src_uda_folder=src_uda_folder,
                             lambda_uda=lambda_uda,
                             uda_train_only_discriminator=uda_train_only_discriminator
                             )
        log.info('Detection training finished!')
        return running_data
    except Exception as e:
        log.exception(str(e))


def inference_detection(inference_folder, training_name,
                        detection_data_classes_field_names=('bounding_boxes_text',),
                        classes_names=('background', 'word'),
                        batch_size=16,
                        gpu_ids='0',
                        padding=0,
                        binary_mask_threshold=0.5,
                        model=None,
                        model_data=None,
                        post_processing=True,
                        post_processing_methods=('size_based_nms',),
                        post_processing_params=None,
                        num_workers=1):
    if not os.path.isdir(os.path.join(DETECTION_MODELS_PATH, training_name)):
        raise ProjectIsNotExists()
    if model is None:
        model = os.path.join(DETECTION_MODELS_PATH, training_name, 'segmentation.pt')
    image_paths_to_lines_and_words = inference(inference_folder, model, detection_data_classes_field_names,
                                               batch_size=batch_size,
                                               depth=6, gpu_ids=gpu_ids, num_workers=num_workers,
                                               binary_mask_lines_threshold=0.99,
                                               binary_mask_words_threshold=binary_mask_threshold, padding=padding)
    # if post_processing:
    #     post_processing_params = post_processing_params if post_processing_params is not None else {}
    #     post_process = DetectionPostProcessing(post_processing_methods, classes_names, **post_processing_params)
    #     image_paths_to_lines_and_words = post_process.post_process(image_paths_to_lines_and_words)
    return image_paths_to_lines_and_words
