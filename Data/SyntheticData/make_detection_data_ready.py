import os
import torch
import json
from torchvision.io import read_image
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from tqdm import tqdm
import torchvision


class ReadyDetectionData:
    def __init__(self, rectangular_bounding_boxes=True, detection_data_classes_field_names=('bounding_boxes_text',), detection_data_fields_to_mask_in_loss=('bounding_boxes_images',), consider_elastic=True, consider_perspective=True):
        self.rectangular_bounding_boxes = rectangular_bounding_boxes
        self.detection_data_classes_field_names = detection_data_classes_field_names
        self.detection_data_fields_to_mask_in_loss = detection_data_fields_to_mask_in_loss
        self.consider_elastic = consider_elastic
        self.consider_perspective = consider_perspective

    def make_detection_data_ready(self, detection_folder_paths):
        for detection_folder_path in tqdm(detection_folder_paths):
            if detection_folder_path.split(os.sep)[-1] == 'detection_data':
                self.__make_detection_folder_data_ready(detection_folder_path)

    def __make_detection_folder_data_ready(self, detection_folder_path):
        data_file_path = os.path.join(detection_folder_path, 'data.json')
        image_data = json.load(open(data_file_path, mode='r', encoding='utf-8'))
        image_data = {
            'bounding_boxes_text_mask_file': image_data['bounding_boxes_text_mask_file'],
            'bounding_boxes_image_mask_file': image_data['bounding_boxes_image_mask_file'],
            'mask_file': image_data['mask_file'],
            'background_file': image_data['background_file'],
            'data_file': data_file_path,
            'image_size': image_data['image_size'],
            'bounding_boxes_text': list(image_data['tags_to_bounding_boxes_text_original'].values()),
            'bounding_boxes_and_text_for_recognition': image_data['bounding_boxes_and_text_for_recognition_original'],
            'bounding_boxes_images': list(image_data['tags_to_bounding_boxes_images_original'].values()),
            'bounding_boxes_images_not_overlapping_text': list(
                image_data['tags_to_bounding_boxes_images_not_overlapping_text_original'].values()),
            'num_pixels_each_class': image_data['num_pixels_each_class'],
            'perspective': image_data['perspective'],
            'elastic': image_data['elastic']
        }
        if self.rectangular_bounding_boxes:
            mask_image = torch.zeros((image_data['image_size'][0], image_data['image_size'][1]))
            for class_idx, field_name in enumerate(self.detection_data_classes_field_names):
                for bounding_box in image_data[field_name]:
                    mask_image[bounding_box['top']: bounding_box['bottom'] + 1, bounding_box['left']: bounding_box['right'] + 1] = class_idx + 1
        else:  # Note : this option is available only when there is only one class in addition to the background class
            mask_image = read_image(image_data['mask_file'], torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0
        for field_name in self.detection_data_fields_to_mask_in_loss:
            for bounding_box in image_data[field_name]:
                mask_image[bounding_box['top']: bounding_box['bottom'] + 1, bounding_box['left']: bounding_box['right'] + 1] = -100
        if self.consider_elastic and image_data['elastic'] is not None:
            mask_image = torch.stack([mask_image, mask_image, mask_image], dim=2)
            indices = np.load(image_data['elastic'])
            mask_image = map_coordinates(mask_image.numpy(), indices, order=1, mode='nearest').reshape(mask_image.shape)[:, :, 0]
            mask_image[mask_image > 0.5] = 1
            mask_image[np.logical_and(mask_image > -50, mask_image <= 0.5).bool()] = 0
            mask_image[mask_image <= -50] = -100
            mask_image = torch.Tensor(mask_image)
        if self.consider_perspective and image_data['perspective'] is not None:
            mask_image = torch.Tensor(cv2.warpPerspective(mask_image.numpy(), np.array(image_data['perspective'][0]), image_data['perspective'][1], flags=cv2.INTER_NEAREST))
        detection_data_ready = mask_image.numpy()
        np.save(os.path.join(detection_folder_path, 'detection_data_ready.npy'), detection_data_ready)

