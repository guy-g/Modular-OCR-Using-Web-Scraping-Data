import os
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np
from Data.dataset_view import LightOCRDatasetView
from Data.conversion_to_coco import convert_detection_to_coco
from Config.setting import *
from Data.compress_datasets import recognition_data_generator
from GeneralUtils.utils import save_to_json


def combine_datasets_clean_failures_split_train_val(combined_dataset_name, dataset_names, combine=True, clean_failures=True, detection=True, recognition=True, split_to_train_validation=False,
                                                    train_split_probability=0.8, split_by_url=True, batch_size=1000000):
    if not split_to_train_validation:
        combined_dataset_dir = os.path.join(DATASETS_PATH, combined_dataset_name)
        os.makedirs(combined_dataset_dir, exist_ok=True)
        combined_dataset_dirs = [combined_dataset_dir]
        combined_dataset_dirs_prob = [1.0]
        train_split_probability = 1.0
    else:
        combined_dataset_train_dir = os.path.join(DATASETS_PATH, combined_dataset_name + '_train')
        os.makedirs(combined_dataset_train_dir, exist_ok=True)
        combined_dataset_val_dir = os.path.join(DATASETS_PATH, combined_dataset_name + '_val')
        os.makedirs(combined_dataset_val_dir, exist_ok=True)
        combined_dataset_dirs = [combined_dataset_train_dir, combined_dataset_val_dir]
        combined_dataset_dirs_prob = [train_split_probability, 1 - train_split_probability]
    combined_samples = 0
    if detection:
        dataview = LightOCRDatasetView('detection', dataset_names, detection=True, recognition=False, to_save=False)
        num_failures = 0
        if split_by_url:
            urls = list(set([img_data['url'] for img_data in dataview.image_paths_to_detection_data.values()]))
            url_to_combined_dataset_dir = {k: np.random.choice(combined_dataset_dirs, p=combined_dataset_dirs_prob) for k in urls}
        for detection_image in tqdm(dataview.image_paths_to_detection_data.keys()):
            detection_dir = os.path.dirname(detection_image)
            if split_by_url:
                combined_dataset_dir = url_to_combined_dataset_dir[dataview.image_paths_to_detection_data[detection_image]['url']]
            else:
                combined_dataset_dir = np.random.choice(combined_dataset_dirs, p=combined_dataset_dirs_prob)
            failure = False
            if clean_failures:
                for fn in os.listdir(detection_dir):
                    file_path = os.path.join(detection_dir, fn)
                    if os.path.getsize(file_path) == 0:
                        print(file_path)
                        failure = True
                        break
                if failure:
                    num_failures += 1
                    shutil.rmtree(detection_dir, ignore_errors=True)
            if combine and not failure:
                sample_dst_dir = os.path.join(combined_dataset_dir, str(combined_samples))
                os.makedirs(sample_dst_dir, exist_ok=True)
                shutil.copytree(detection_dir, os.path.join(sample_dst_dir, 'detection_data'))
                combined_samples += 1
        print('Num detection failures: {}. All failures deleted.'.format(num_failures))
    if recognition:
        num_failures = 0
        num_whole_folders_failures = 0
        dataview = LightOCRDatasetView('recognition', dataset_names, detection=False, recognition=True, to_save=False)
        recognition_dirs_failed = set()
        recognition_dirs = set()
        for recognition_image in tqdm(dataview.image_paths_to_recognition_data.keys()):
            if os.path.getsize(recognition_image) == 0:
                num_failures += 1
                os.remove(recognition_image)
            if not os.path.isfile(os.path.join(os.path.dirname(recognition_image), 'data.json')):
                num_whole_folders_failures += 1
                shutil.rmtree(os.path.dirname(recognition_image), ignore_errors=True)
                recognition_dirs_failed.add(os.path.dirname(recognition_image))
            recognition_dirs.add(os.path.dirname(recognition_image))
        print('Num recognition image failures: {}. All failures deleted.'.format(num_failures))
        print('Num recognition folders failures: {}. All failures deleted.'.format(num_whole_folders_failures))
        if combine:
            for part, image_paths_to_recognition_data in enumerate(recognition_data_generator(dataset_names, batch_size)):
                image_paths = list(image_paths_to_recognition_data.keys())
                train_indexes = np.random.choice(list(range(len(image_paths))), size=int(train_split_probability * len(image_paths)), replace=False)
                rec_data_dir = os.path.join(combined_dataset_dirs[0], str(combined_samples))
                os.makedirs(rec_data_dir)
                rec_data_dir = os.path.join(rec_data_dir, 'recognition_data')
                os.makedirs(rec_data_dir)
                data_dict = {}
                for idx, ik in enumerate(train_indexes):
                    dst_path = os.path.join(rec_data_dir, '{}.{}'.format(idx, image_paths[ik].split('.')[-1]))
                    shutil.copyfile(image_paths[ik], dst_path)
                    data_dict['{}.{}'.format(idx, image_paths[ik].split('.')[-1])] = image_paths_to_recognition_data[image_paths[ik]]
                combined_samples += 1
                save_to_json(data_dict, os.path.join(rec_data_dir, 'data.json'))
                if len(combined_dataset_dirs) > 1:
                    rec_data_dir = os.path.join(combined_dataset_dirs[1], str(combined_samples))
                    os.makedirs(rec_data_dir)
                    rec_data_dir = os.path.join(rec_data_dir, 'recognition_data')
                    os.makedirs(rec_data_dir)
                    data_dict = {}
                    val_indexes = [i for i in range(len(image_paths)) if i not in train_indexes]
                    for idx, ik in enumerate(val_indexes):
                        dst_path = os.path.join(rec_data_dir, '{}.{}'.format(idx, image_paths[ik].split('.')[-1]))
                        shutil.copyfile(image_paths[ik], dst_path)
                        data_dict['{}.{}'.format(idx, image_paths[ik].split('.')[-1])] = image_paths_to_recognition_data[image_paths[ik]]
                    combined_samples += 1
                    save_to_json(data_dict, os.path.join(rec_data_dir, 'data.json'))
    for combined_dataset_dir in combined_dataset_dirs:
        convert_detection_to_coco(combined_dataset_dir)


if __name__ == '__main__':
    combine_datasets_clean_failures_split_train_val('clean', ['clean_train'])


