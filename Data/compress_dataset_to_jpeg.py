from PIL import Image
from tqdm import tqdm
import os
import json
from Data.dataset_view import OCRDatasetView
from Config.setting import *
from GeneralUtils.utils import *


def compress_dataset(dataset_names,
                     language_name=None,
                     max_word_length=None,
                     file_names_to_compress_detection=('image', 'background', 'background_images', 'demonstration'),
                     erase_elastic_data=False, enable_icon_or_emoji_text=False):
    if dataset_names == 'all':
        dataset_names = [fn for fn in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, fn))]
    for dataset_name in dataset_names:
        dataset_folder = os.path.join(DATASETS_PATH, dataset_name)
        for (parent_dir, _, file_names) in tqdm(os.walk(dataset_folder)):
            try:
                if parent_dir.split(os.sep)[-1] == 'detection_data':
                    data_file = os.path.join(parent_dir, 'data.json')
                    data_coco_file = os.path.join(parent_dir, 'detection_data_coco.json')
                    if file_names_to_compress_detection is not None:
                        for file_name in file_names_to_compress_detection:
                            if file_name + '.png' in file_names:
                                tmp = Image.open(os.path.join(parent_dir, file_name + '.png')).convert('RGB')
                                tmp.save(os.path.join(parent_dir, file_name + '.jpg'))
                                os.remove(os.path.join(parent_dir, file_name + '.png'))
                    else:
                        for file_name in file_names:
                            if file_name.endswith(('png',)):
                                tmp = Image.open(os.path.join(parent_dir, file_name)).convert('RGB')
                                tmp.save(os.path.join(parent_dir, file_name[:-4] + '.jpg'))
                                os.remove(os.path.join(parent_dir, file_name))
                    if os.path.isfile(data_file) and (file_names_to_compress_detection is None or 'background' in file_names_to_compress_detection):
                        image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
                        image_data["background_file"] = os.path.join(parent_dir, "background.jpg")
                        save_to_json(image_data, data_file)
                    if os.path.isfile(data_file) and (file_names_to_compress_detection is None or 'background_images' in file_names_to_compress_detection):
                        image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
                        image_data["background_images_file"] = os.path.join(parent_dir, "background_images.jpg")
                        save_to_json(image_data, data_file)
                    if os.path.isfile(data_coco_file) and (file_names_to_compress_detection is None or 'image' in file_names_to_compress_detection):
                        image_data = json.load(open(data_coco_file, mode='r', encoding='utf-8'))
                        image_data["images"][0]["file_name"] = "image.jpg"
                        save_to_json(image_data, data_coco_file)
                    if erase_elastic_data and os.path.isfile(os.path.join(parent_dir, 'indices.npy')):
                        os.remove(os.path.join(parent_dir, 'indices.npy'))
                elif parent_dir.split(os.sep)[-1] == 'recognition_data':
                    data_file = os.path.join(parent_dir, 'data.json')
                    image_data = json.load(open(data_file, mode='r', encoding='utf-8'))
                    for file_name in file_names:
                        if file_name.endswith(('png',)) and '_mask' not in file_name:
                            tmp = Image.open(os.path.join(parent_dir, file_name)).convert('RGB')
                            tmp.save(os.path.join(parent_dir, file_name[:-4] + '.jpg'))
                            os.remove(os.path.join(parent_dir, file_name))
                            if file_name in image_data.keys():
                                image_data[file_name[:-4] + '.jpg'] = image_data[file_name]
                                del image_data[file_name]
                                # image_data[file_name[:-4] + '.jpg']["mask_file"] = os.path.join(parent_dir, file_name[:-4] + '_mask.jpg')  # WE DONT NEED TO COMPRESS MASKS!
                    save_to_json(image_data, data_file)
                elif parent_dir.split(os.sep)[-1] == 'layout_data':
                    if file_names_to_compress_detection is not None:
                        for file_name in file_names_to_compress_detection:
                            if file_name + '.png' in file_names:
                                tmp = Image.open(os.path.join(parent_dir, file_name + '.png')).convert('RGB')
                                tmp.save(os.path.join(parent_dir, file_name + '.jpg'))
                                os.remove(os.path.join(parent_dir, file_name + '.png'))
                    else:
                        for file_name in file_names:
                            if file_name.endswith(('png',)):
                                tmp = Image.open(os.path.join(parent_dir, file_name)).convert('RGB')
                                tmp.save(os.path.join(parent_dir, file_name[:-4] + '.jpg'))
                                os.remove(os.path.join(parent_dir, file_name))
                if 'demonstration.png' in file_names:  # os.path.isdir(os.path.join(parent_dir, 'detection_data')) and os.path.isdir(os.path.join(parent_dir, 'recognition_data')) and
                    tmp = Image.open(os.path.join(parent_dir, 'demonstration.png')).convert('RGB')
                    tmp.save(os.path.join(parent_dir, 'demonstration.jpg'))
                    os.remove(os.path.join(parent_dir, 'demonstration.png'))
                if 'demonstration_layout.png' in file_names:  # os.path.isdir(os.path.join(parent_dir, 'detection_data')) and os.path.isdir(os.path.join(parent_dir, 'recognition_data')) and
                    tmp = Image.open(os.path.join(parent_dir, 'demonstration_layout.png')).convert('RGB')
                    tmp.save(os.path.join(parent_dir, 'demonstration_layout.jpg'))
                    os.remove(os.path.join(parent_dir, 'demonstration_layout.png'))
            except Exception as e:
                print(str(e))
        # OCRDatasetView(dataset_name,
        #                DATASETS_VIEWS_PATH,
        #                DATASETS_PATH,
        #                load_view_file=False, inference=False,
        #                running_names=[dataset_name],
        #                language_name=language_name,
        #                language_folder=LANGUAGES_PATH,
        #                max_word_length=max_word_length,
        #                detection=True,
        #                recognition=True,
        #                layout=False,
        #                enable_icon_or_emoji_text=enable_icon_or_emoji_text)


if __name__ == '__main__':
    compress_dataset(
        'all',
        None
    )

