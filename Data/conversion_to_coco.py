import os
import json
from GeneralUtils.utils import *


def convert_detection_to_coco(folder_path):
    coco_format_whole_dataset = {"categories": None, "images": [], "annotations": []}
    for (p, _, child_files) in os.walk(folder_path):
        if p.split(os.sep)[-1] == 'detection_data' and os.path.isdir(p) and os.path.isfile(os.path.join(p, 'data.json')):
            if 'image.png' in child_files:
                ext = 'png'
            else:
                ext = 'jpg'
            detection_data = json.load(open(os.path.join(p, 'data.json'), mode='r', encoding='utf-8'))
            [h, w] = detection_data['image_size']
            coco_format = {
                "categories": [
                    {
                        "id": 1,
                        "name": "word",
                        "supercategory": "text"
                    },
                    {
                        "id": 2,
                        "name": "image",
                        "supercategory": "figure"
                    }
                ],
                "images": [
                    {
                        "id": len(coco_format_whole_dataset['images']) + 1,
                        "file_name": 'image.' + ext,
                        "height": h,
                        "width": w,
                        "date_captured": "2013-11-18 02:53:27"
                    }
                ],
                "annotations": [
                    {
                        "id": len(coco_format_whole_dataset['annotations']) + ibb + 1,
                        "image_id": len(coco_format_whole_dataset['images']) + 1,
                        "category_id": 1,
                        "segmentation": [[bb['left'], bb['top'], bb['left'], bb['bottom'], bb['right'], bb['bottom'], bb['right'], bb['top']]],
                        "bbox": [bb['left'], bb['top'], bb['right'] - bb['left'], bb['bottom'] - bb['top']],
                        "text": detection_data["tags_to_text"][tag]
                    } for ibb, (tag, bb) in enumerate(detection_data['tags_to_bounding_boxes_text'].items())
                ]
            }
            coco_format['annotations'] += [
                {
                    "id": len(coco_format_whole_dataset['annotations']) + len(coco_format['annotations']) + ibb + 1,
                    "image_id": len(coco_format_whole_dataset['images']) + 1,
                    "category_id": 2,
                    "segmentation": [[bb['left'], bb['top'], bb['left'], bb['bottom'], bb['right'], bb['bottom'], bb['right'], bb['top']]],
                    "bbox": [bb['left'], bb['top'], bb['right'] - bb['left'], bb['bottom'] - bb['top']],
                    "text": None
                } for ibb, bb in enumerate(detection_data['tags_to_bounding_boxes_images'].values())
            ]
            coco_format_whole_dataset['categories'] = coco_format["categories"]
            coco_format_whole_dataset['annotations'] += coco_format["annotations"]
            coco_format_whole_dataset['images'] += coco_format["images"]
            coco_format_whole_dataset['images'][-1]["file_name"] = os.path.join(p, 'image.' + ext)
            save_to_json(coco_format, os.path.join(p, 'detection_data_coco.json'))
    save_to_json(coco_format_whole_dataset, os.path.join(folder_path, 'detection_dataset_coco.json'))


def convert_layout_to_coco(folder_path):
    coco_format_whole_dataset = {"categories": None, "images": [], "annotations": []}
    for (p, _, child_files) in os.walk(folder_path):
        if p.split(os.sep)[-1] == 'layout_data' and os.path.isdir(p) and os.path.isfile(os.path.join(p, 'data.json')):
            if 'image.png' in child_files:
                ext = 'png'
            else:
                ext = 'jpg'
            layout_data = json.load(open(os.path.join(p, 'data.json'), mode='r', encoding='utf-8'))
            [h, w] = layout_data['image_size']
            coco_format = {
                "categories": [
                    {
                        "id": 1,
                        "name": "paragraph",
                        "supercategory": "text"
                    }
                ],
                "images": [
                    {
                        "id": len(coco_format_whole_dataset['images']) + 1,
                        "file_name": 'image.' + ext,
                        "height": h,
                        "width": w,
                        "date_captured": "2013-11-18 02:53:27"
                    }
                ],
                "annotations": [
                    {
                        "id": len(coco_format_whole_dataset['annotations']) + ibb + 1,
                        "image_id": len(coco_format_whole_dataset['images']) + 1,
                        "category_id": 1,
                        "segmentation": [[paragraph['left'], paragraph['top'], paragraph['left'], paragraph['bottom'], paragraph['right'], paragraph['bottom'], paragraph['right'], paragraph['top']]],
                        "bbox": [paragraph['left'], paragraph['top'], paragraph['right'] - paragraph['left'], paragraph['bottom'] - paragraph['top']]
                    } for ibb, (paragraph, words) in enumerate(layout_data['paragraphs'])
                ]
            }
            coco_format_whole_dataset['categories'] = coco_format["categories"]
            coco_format_whole_dataset['annotations'] += coco_format["annotations"]
            coco_format_whole_dataset['images'] += coco_format["images"]
            coco_format_whole_dataset['images'][-1]["file_name"] = os.path.join(p, 'image.' + ext)
            save_to_json(coco_format, os.path.join(p, 'layout_data_coco.json'))
    save_to_json(coco_format_whole_dataset, os.path.join(folder_path, 'layout_dataset_coco.json'))

