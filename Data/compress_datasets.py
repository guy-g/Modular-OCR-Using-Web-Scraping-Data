from PIL import Image
from tqdm import tqdm
import os
import json
import zipfile, glob
import shutil
from Data.dataset_view import OCRDatasetView, LightOCRDatasetView, recognition_data_generator
from Config.setting import DATASETS_PATH, DATASETS_VIEWS_PATH, LANGUAGES_PATH
from torchvision.io import read_image
import torchvision
from Recognition.recognition import resize_and_norm_image
from GeneralUtils.utils import *
import numpy as np
from GeneralUtils.project_logs import create_log
from zipfile import ZipFile


def prepare_recognition_zip_for_external_training(zip_prefix_name, dataset_names, batch_size=1000000):
    for part, image_paths_to_recognition_data in enumerate(recognition_data_generator(dataset_names, batch_size)):
        working_dir = make_tmp_folder()
        rec_dir = os.path.join(working_dir, '{}_recognition_dataset_part_{}'.format(zip_prefix_name, part + 1))
        os.makedirs(rec_dir, exist_ok=True)
        rec_dir = os.path.join(rec_dir, 'recognition_data')
        os.makedirs(rec_dir, exist_ok=True)
        new_rec_data = {}
        for ik, (k, v) in tqdm(enumerate(image_paths_to_recognition_data.items())):
            fn = k.split(os.sep)[-1]
            pre, ext = fn.split('.')[0], fn.split('.')[-1]
            sample_file = '{}.{}'.format(ik, ext)
            shutil.copyfile(k, os.path.join(rec_dir, sample_file))
            new_rec_data[sample_file] = v
        save_to_json(new_rec_data, os.path.join(rec_dir, 'data.json'))
        shutil.make_archive(os.path.join(DATASETS_PATH, '{}_recognition_data_part_{}'.format(zip_prefix_name, part + 1)), 'zip', working_dir, logger=create_log('zipping', TMP_PATH), verbose=True)
        shutil.rmtree(working_dir, ignore_errors=True)


def validate_recognition_zips_with_jpg(zip_file_paths):
    for zip_file in zip_file_paths:
        with ZipFile(zip_file, 'r') as zipObj:
            listOfiles = zipObj.namelist()
            num_png = 0
            for elem in listOfiles:
                if 'png' in elem:
                    num_png += 1
            print('{} : {} / {} files are png'.format(zip_file.split(os.sep)[-1], num_png, len(listOfiles)))


def prepare_recognition_zip_for_external_training_old(zip_prefix_name, dataset_names, partitions=1):
    dataview = OCRDatasetView('recognition_to_zip', DATASETS_VIEWS_PATH, DATASETS_PATH, running_names=dataset_names, detection=False, recognition=True, layout=False, to_save=False)
    file_names = list(dataview.image_paths_to_recognition_data.keys())
    partition_size = int(np.ceil(len(file_names) / float(partitions)))
    print('Num samples : {}\nDividing to {} partitions, each of size {}'.format(len(file_names), partitions, partition_size))
    for part in range(partitions):
        working_dir = make_tmp_folder()
        rec_dir = os.path.join(rec_dir, 'recognition_data')
        os.makedirs(rec_dir, exist_ok=True)
        new_rec_data = {}
        for ik, k in tqdm(enumerate(file_names[part * partition_size: (part + 1) * partition_size])):
            fn = k.split(os.sep)[-1]
            pre, ext = fn.split('.')[0], fn.split('.')[-1]
            sample_file = '{}.{}'.format(ik, ext)
            shutil.copyfile(k, os.path.join(rec_dir, sample_file))
            new_rec_data[sample_file] = dataview.image_paths_to_recognition_data[k]
        save_to_json(new_rec_data, os.path.join(rec_dir, 'data.json'))
        shutil.make_archive(os.path.join(DATASETS_PATH, '{}_recognition_data_part_{}'.format(zip_prefix_name, part + 1)), 'zip', working_dir)
        shutil.rmtree(working_dir, ignore_errors=True)


def prepare_detection_zip_for_external_training(zip_prefix_name, dataset_names, partitions=1):
    dataview = OCRDatasetView('detection_to_zip', DATASETS_VIEWS_PATH, DATASETS_PATH, running_names=dataset_names, detection=True, recognition=False, layout=False, to_save=False, enable_icon_or_emoji_text=True)
    file_names = list(dataview.image_paths_to_detection_data.keys())
    partition_size = int(np.ceil(len(file_names) / float(partitions)))
    print('Num samples : {}\nDividing to {} partitions, each of size {}'.format(len(file_names), partitions, partition_size))
    for part in range(partitions):
        working_dir = make_tmp_folder()
        det_dir = os.path.join(working_dir, '{}_detection_dataset_part_{}'.format(zip_prefix_name, part + 1))
        os.makedirs(det_dir, exist_ok=True)
        for ik, k in tqdm(enumerate(file_names[part * partition_size: (part + 1) * partition_size])):
            sample_dir_dst = os.path.join(det_dir, str(ik))
            os.makedirs(sample_dir_dst, exist_ok=True)
            sample_dir_dst = os.path.join(sample_dir_dst, 'detection_data')
            os.makedirs(sample_dir_dst, exist_ok=True)
            sample_dir = '{}'.format(os.sep).join(k.split(os.sep)[:-1])
            shutil.copyfile(k, os.path.join(sample_dir_dst, k.split(os.sep)[-1]))
            shutil.copyfile(os.path.join(sample_dir, 'bounding_boxes_words_mask_transformed.png'), os.path.join(sample_dir_dst, 'bounding_boxes_words_mask_transformed.png'))
            shutil.copyfile(os.path.join(sample_dir, 'bounding_boxes_images_mask_transformed.png'), os.path.join(sample_dir_dst, 'bounding_boxes_images_mask_transformed.png'))
            shutil.copyfile(os.path.join(sample_dir, 'data.json'), os.path.join(sample_dir_dst, 'data.json'))
        shutil.make_archive(os.path.join(DATASETS_PATH, '{}_detection_data_part_{}'.format(zip_prefix_name, part + 1)), 'zip', working_dir)
        shutil.rmtree(working_dir, ignore_errors=True)


def prepare_recognition_data_for_faster_training(dataset_folder,
                                                 image_size,
                                                 recognition_resize_method):
    norm = torchvision.transforms.Normalize(0, 1)
    resize_transformation_without_ar = torchvision.transforms.Resize(image_size)
    for (parent_dir, _, file_names) in tqdm(os.walk(dataset_folder)):
        if parent_dir.split(os.sep)[-1] == 'recognition_data':
            data_file = os.path.join(parent_dir, 'data.json')
            try:
                image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
                for file_name in file_names:
                    if file_name.lower().endswith(OCRDatasetView.IMAGES_EXT):
                        image_path = os.path.join(parent_dir, file_name)
                        image = read_image(image_path, torchvision.io.ImageReadMode.RGB)
                        image = resize_and_norm_image(image, recognition_resize_method, image_size, resize_transformation_without_ar, norm, True)
                        torchvision.utils.save_image(image, image_path)
                for k in image_data.keys():
                    image_data[k]['image_size'] = [image_size[0], image_size[1]]
                save_to_json(image_data, data_file)
            except Exception as e:
                print(str(e))

