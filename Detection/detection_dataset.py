import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data.dataloader import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from GeneralUtils.utils import *
from Detection.model import *


class DetectionDataset(Dataset):
    def __init__(self, image_paths_to_detection_data, detection_data_classes_field_names, detection_data_fields_to_mask_in_loss=None,
                 rectangular_bounding_boxes=True, crop=(320, 320), depth=6, inference=False, consider_perspective=True, consider_elastic=True, detection_data_ready=False,
                 mask_in_loss_first=True, random_transformation=0.25, duplicate_data_to=None):
        self.image_paths_to_detection_data = image_paths_to_detection_data
        self.detection_data_classes_field_names = detection_data_classes_field_names
        self.detection_data_fields_to_mask_in_loss = [] if detection_data_fields_to_mask_in_loss is None else detection_data_fields_to_mask_in_loss
        self.rectangular_bounding_boxes = rectangular_bounding_boxes
        self.image_paths = list(self.image_paths_to_detection_data.keys())
        if duplicate_data_to is not None:
            while duplicate_data_to > len(self.image_paths):
                self.image_paths += list(self.image_paths_to_detection_data.keys())
        self.crop = crop
        self.depth = depth
        self.inference = inference
        self.consider_perspective = consider_perspective
        self.consider_elastic = consider_elastic
        self.detection_data_ready = detection_data_ready
        self.mask_in_loss_first = mask_in_loss_first
        self.random_transformation = random_transformation
        self.norm = torchvision.transforms.Normalize(0, 1)
        self.transformations = [
            torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
            torchvision.transforms.GaussianBlur((7, 7), (0.1, 2)),
            torchvision.transforms.RandomPosterize(bits=1, p=1.0),
            torchvision.transforms.RandomSolarize(120, p=1.0),
            torchvision.transforms.RandomInvert(p=1.0)
        ]
        self.random_rotations = [
            torchvision.transforms.RandomRotation((90, 90)),
            torchvision.transforms.RandomRotation((180, 180)),
            torchvision.transforms.RandomRotation((270, 270))
        ]
        if self.crop is not None:
            self.pad_to_h = compute_possible_closest_shape(value=crop[0], depth=depth)
            self.pad_to_w = compute_possible_closest_shape(value=crop[1], depth=depth)
            self.resize_transform = torchvision.transforms.Resize(self.crop)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        (h_src, w_src) = image.shape[1], image.shape[2]
        if not self.inference:
            if not self.detection_data_ready:
                image_data = json.load(open(self.image_paths_to_detection_data[image_path]['data_file'], mode='r', encoding='utf-8'))
                image_data = {
                    'bounding_boxes_text_mask_file': image_data['bounding_boxes_text_mask_file'],
                    'bounding_boxes_image_mask_file': image_data['bounding_boxes_image_mask_file'],
                    'mask_file': image_data['mask_file'],
                    'background_file': image_data['background_file'],
                    'data_file': self.image_paths_to_detection_data[image_path]['data_file'],
                    'image_size': image_data['image_size'],
                    'bounding_boxes_text': list(image_data['tags_to_bounding_boxes_text_original'].values()),
                    'bounding_boxes_and_text_for_recognition': image_data['bounding_boxes_and_text_for_recognition_original'],
                    'bounding_boxes_images': list(image_data['tags_to_bounding_boxes_images_original'].values()),
                    'bounding_boxes_images_not_overlapping_text': list(image_data['tags_to_bounding_boxes_images_not_overlapping_text_original'].values()),
                    'num_pixels_each_class': image_data['num_pixels_each_class'],
                    'perspective': image_data['perspective'],
                    'elastic': image_data['elastic']
                }
                mask_image = torch.zeros((image.shape[1], image.shape[2]))
                if not self.mask_in_loss_first:
                    if self.rectangular_bounding_boxes:
                        for class_idx, field_name in enumerate(self.detection_data_classes_field_names):
                            for bounding_box in image_data[field_name]:
                                mask_image[bounding_box['top']: bounding_box['bottom'] + 1, bounding_box['left']: bounding_box['right'] + 1] = class_idx + 1
                    else:  # Note : this option is available only when there is only one class in addition to the background class
                        mask_image = read_image(image_data['mask_file'], torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0
                    for field_name in self.detection_data_fields_to_mask_in_loss:
                        for bounding_box in image_data[field_name]:
                            mask_image[bounding_box['top']: bounding_box['bottom'] + 1, bounding_box['left']: bounding_box['right'] + 1] = -100
                else:
                    for field_name in self.detection_data_fields_to_mask_in_loss:
                        for bounding_box in image_data[field_name]:
                            mask_image[bounding_box['top']: bounding_box['bottom'] + 1, bounding_box['left']: bounding_box['right'] + 1] = -100
                    if self.rectangular_bounding_boxes:
                        for class_idx, field_name in enumerate(self.detection_data_classes_field_names):
                            for bounding_box in image_data[field_name]:
                                mask_image[bounding_box['top']: bounding_box['bottom'] + 1, bounding_box['left']: bounding_box['right'] + 1] = class_idx + 1
                    else:  # Note : this option is available only when there is only one class in addition to the background class
                        tmp = read_image(image_data['mask_file'], torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0
                        mask_image[tmp > 0] = tmp[tmp > 0]
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
            elif self.rectangular_bounding_boxes:
                mask_image = torch.zeros((image.shape[1], image.shape[2]))
                if not self.mask_in_loss_first:
                    for class_idx, field_name in enumerate(self.detection_data_classes_field_names):
                        tmp = (read_image(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), field_name), torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0) * (1 + class_idx)
                        mask_image[tmp == (1 + class_idx)] = 1 + class_idx
                    for field_name in self.detection_data_fields_to_mask_in_loss:
                        tmp = (read_image(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), field_name), torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0) * (-100)
                        mask_image[tmp == -100] = -100
                else:
                    for field_name in self.detection_data_fields_to_mask_in_loss:
                        tmp = (read_image(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), field_name), torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0) * (-100)
                        mask_image[tmp == -100] = -100
                    for class_idx, field_name in enumerate(self.detection_data_classes_field_names):
                        tmp = (read_image(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), field_name), torchvision.io.ImageReadMode.RGB)[0, :, :] / 255.0) * (1 + class_idx)
                        mask_image[tmp == (1 + class_idx)] = 1 + class_idx
        (h, w) = image.shape[1], image.shape[2]
        if self.crop is not None:
            if np.random.rand() < self.random_transformation:
                h_to_crop = np.random.randint(max(int(0.75 * self.crop[0]), 1), min(int(1.25 * self.crop[0]), h))
                w_to_crop = np.random.randint(max(int(0.75 * self.crop[1]), 1), min(int(1.25 * self.crop[1]), w))
                crop_by = (h_to_crop, w_to_crop)
                i, j, h_to_crop, w_to_crop = torchvision.transforms.RandomCrop.get_params(image, crop_by)
                image = self.resize_transform(image[:, i: i + h_to_crop, j: j + w_to_crop])
                if not self.inference:
                    mask_image = self.resize_transform(mask_image[i: i + h_to_crop, j: j + w_to_crop].view((1, h_to_crop, w_to_crop)))
                    mask_image = mask_image.view((mask_image.shape[1], mask_image.shape[2]))
                    mask_image[mask_image > 0.5] = 1
                    mask_image[np.logical_and(mask_image > -50, mask_image <= 0.5).bool()] = 0
                    mask_image[mask_image <= -50] = -100
                _, h, w = image.shape
            else:
                crop_by = (min(h, self.crop[0]), min(w, self.crop[1]))
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(image, crop_by)
                image = image[:, i: i + h, j: j + w]
                if not self.inference:
                    mask_image = mask_image[i: i + h, j: j + w]
        else:
            self.pad_to_h, self.pad_to_w = compute_possible_closest_shape(value=h, depth=self.depth), compute_possible_closest_shape(value=w, depth=self.depth)
        if self.random_transformation > 0.0:
            for transform in self.transformations:
                if np.random.rand() < self.random_transformation:
                    image = transform(image)
            if np.random.rand() < self.random_transformation:
                rot = np.random.randint(0, 3)
                image = self.random_rotations[rot](image)
                if not self.inference:
                    mask_image = self.random_rotations[rot](mask_image.view(1, mask_image.shape[0], mask_image.shape[1]))[0, :, :]
        image = image / 255.0
        if h != self.pad_to_h or w != self.pad_to_w:
            image = F.pad(image, (0, self.pad_to_w - w, 0, self.pad_to_h - h))
            if not self.inference:
                mask_image = F.pad(mask_image, (0, self.pad_to_w - w, 0, self.pad_to_h - h))
        image = self.norm(image)
        if not self.inference:
            return image, mask_image.long(), image_path
        return image, image_path, (h_src, w_src)

    def __len__(self):
        return len(self.image_paths)

