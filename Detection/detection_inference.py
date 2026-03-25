import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from Data.dataset_view import OCRDatasetView
from ModelUtils.model_utils import *
import numpy as np
from skimage.measure import label, regionprops
import cv2
from GeneralUtils.utils import *
from Detection.model import *
from Detection.detection_dataset import DetectionDataset
from Data.SyntheticData.euclidean_utils import vertical_1D_iou
from Detection.lines import lines_and_words_extraction


@torch.no_grad()
def inference(inference_folder, model, detection_data_classes_field_names, batch_size=1, depth=6, gpu_ids='0', num_workers=1,
              binary_mask_lines_threshold=0.99, binary_mask_words_threshold=0.95, padding=0):
    inference_dataset_view = OCRDatasetView('inference', inference_folder, inference_folder, inference=True, detection=True, recognition=False, layout=False, to_save=False)
    device = get_device(gpu_ids)
    segmentation_model = load_model_by_gpu_ids(model, gpu_ids)
    inference_dataset = DetectionDataset(inference_dataset_view.image_paths_to_detection_data, detection_data_classes_field_names, crop=None, depth=depth, inference=True, random_transformation=0.0)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    image_paths_to_lines_and_words = {}
    for images, paths, (H, W) in inference_dataloader:
        images = images.to(device)
        preds = inference_model(segmentation_model, images)
        preds = F.softmax(preds, dim=1)
        for ipred, pred in enumerate(preds):
            lines_and_words = lines_and_words_extraction(preds, H.item(), W.item(),
                                                                     binary_mask_lines_threshold=binary_mask_lines_threshold,
                                                                     binary_mask_words_threshold=binary_mask_words_threshold,
                                                                     padding=padding)
            image_paths_to_lines_and_words[paths[ipred]] = lines_and_words
    return image_paths_to_lines_and_words


@torch.no_grad()
def inference_discriminator(inference_folder, model, detection_data_classes_field_names, batch_size=1, depth=6, gpu_ids='0', num_workers=1,
              binary_mask_lines_threshold=0.99, binary_mask_words_threshold=0.99, padding=0):
    inference_dataset_view = OCRDatasetView('inference', inference_folder, inference_folder, inference=True, detection=True, recognition=False, layout=False, to_save=False)
    device = get_device(gpu_ids)
    segmentation_model = load_model_by_gpu_ids(model, gpu_ids)
    inference_dataset = DetectionDataset(inference_dataset_view.image_paths_to_detection_data, detection_data_classes_field_names, crop=None, depth=depth, inference=True, random_transformation=0.0)
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    avg_disc_score = 0.0
    for images, paths, (H, W) in inference_dataloader:
        images = images.to(device)
        preds = segmentation_model(images, images)
        disc_preds = F.softmax(preds['discriminator_unlabeled_preds'], dim=1)
        print('gg')
    return avg_disc_score


def split_lines(lines_and_words_bbs):
    lines_and_words_bbs_splitted = []
    for idx_line, line in enumerate(lines_and_words_bbs):
        bounding_boxes_words_splitted_into_sublines = []
        for ibb_word, bb_word_rel in enumerate(line['words']):
            found_subline = False
            for isubline, subline in enumerate(bounding_boxes_words_splitted_into_sublines):
                subline_words = subline['words']
                for word in subline_words:
                    if vertical_1D_iou(word['bounding_box'], bb_word_rel['bounding_box']) > 0.7:
                        bounding_boxes_words_splitted_into_sublines[isubline]['words'].append(bb_word_rel)
                        found_subline = True
                        break
            if not found_subline:
                bounding_boxes_words_splitted_into_sublines.append({'line_bounding_box': [], 'words': [bb_word_rel]})
        for isubline in range(len(bounding_boxes_words_splitted_into_sublines)):
            bounding_boxes_words_splitted_into_sublines[isubline]['line_bounding_box'] = [
                min([i['bounding_box'][0] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']]),
                min([i['bounding_box'][1] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']]),
                max([i['bounding_box'][2] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']]),
                max([i['bounding_box'][3] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']])
            ]
        lines_and_words_bbs_splitted += bounding_boxes_words_splitted_into_sublines
    return lines_and_words_bbs_splitted


def segmentation_output_to_lines_and_words(segmentation_output, H, W, binary_mask_lines_threshold=0.99, binary_mask_words_threshold=0.99, padding=0):
    binary_map = ((segmentation_output.detach().cpu()[1, :, :] >= 0.9) * 1).numpy()
    blurred_binary_map = (cv2.blur(binary_map.astype(np.float_), (21, 1)) > 0.001) * 1
    lines_binary_mask = (blurred_binary_map >= binary_mask_lines_threshold) * 1
    connected_components_lines = label(lines_binary_mask)
    lines_bounding_boxes = [
        [
            i.bbox[1],
            i.bbox[0],
            i.bbox[3] - 1,
            i.bbox[2] - 1
        ] for i in regionprops(connected_components_lines) if i.bbox[0] < (i.bbox[2] - 1) and i.bbox[1] < (i.bbox[3] - 1)
    ]
    lines_and_words_bounding_boxes = []
    for line_bb in lines_bounding_boxes:
        line_segmentation_output = segmentation_output[1, line_bb[1]: line_bb[3] + 1, line_bb[0]: line_bb[2] + 1]
        words_binary_mask = (line_segmentation_output.detach().cpu() >= binary_mask_words_threshold) * 1
        connected_components_words = label(words_binary_mask)
        words_bounding_boxes = [
            [
                i.bbox[1],
                i.bbox[0],
                i.bbox[3] - 1,
                i.bbox[2] - 1
            ] for i in regionprops(connected_components_words) if i.bbox[0] < (i.bbox[2] - 1) and i.bbox[1] < (i.bbox[3] - 1)
        ]
        words_bounding_boxes.sort(key=lambda x: x[0])
        splitted_lines_and_words_bounding_boxes = []
        for iword_bb, word_bb in enumerate(words_bounding_boxes):
            word_bb_rel = [
                max(word_bb[0] + line_bb[0] - padding, 0),
                max(word_bb[1] + line_bb[1] - padding, 0),
                min(word_bb[2] + line_bb[0] + padding, W - 1),
                min(word_bb[3] + line_bb[1] + padding, H - 1)
            ]
            found_splitted_line = False
            for isplitted_line, splitted_line in enumerate(splitted_lines_and_words_bounding_boxes):
                splitted_line_words = splitted_line['words']
                for word in splitted_line_words:
                    word_bb = word['bounding_box']
                    if vertical_1D_iou(word_bb, word_bb_rel) > 0.1:
                        splitted_lines_and_words_bounding_boxes[isplitted_line]['words'].append({'bounding_box': word_bb_rel})
                        found_splitted_line = True
                        break
            if not found_splitted_line:
                splitted_lines_and_words_bounding_boxes.append({'line_bounding_box': [], 'words': [{'bounding_box': word_bb_rel}]})
        for isplitted_line in range(len(splitted_lines_and_words_bounding_boxes)):
            splitted_lines_and_words_bounding_boxes[isplitted_line]['line_bounding_box'] = [
                min([i['bounding_box'][0] for i in splitted_lines_and_words_bounding_boxes[isplitted_line]['words']]),
                min([i['bounding_box'][1] for i in splitted_lines_and_words_bounding_boxes[isplitted_line]['words']]),
                max([i['bounding_box'][2] for i in splitted_lines_and_words_bounding_boxes[isplitted_line]['words']]),
                max([i['bounding_box'][3] for i in splitted_lines_and_words_bounding_boxes[isplitted_line]['words']])
            ]
        lines_and_words_bounding_boxes += splitted_lines_and_words_bounding_boxes
    return lines_and_words_bounding_boxes


def segmentation_output_to_words(segmentation_output, H, W, binary_mask_lines_threshold=0.99, binary_mask_words_threshold=0.99, padding=0):
    binary_map = ((segmentation_output.detach().cpu()[1, :, :] >= binary_mask_words_threshold) * 1).numpy()
    connected_components_lines = label(binary_map)
    lines_bounding_boxes = [
        [
            i.bbox[1],
            i.bbox[0],
            i.bbox[3] - 1,
            i.bbox[2] - 1
        ] for i in regionprops(connected_components_lines) if i.bbox[0] < (i.bbox[2] - 1) and i.bbox[1] < (i.bbox[3] - 1)
    ]
    lines_and_words_bounding_boxes = []
    for line_bb in lines_bounding_boxes:
        lines_and_words_bounding_boxes.append({
            'line_bounding_box': line_bb, 'words': [{'bounding_box': line_bb}]
        })
    return lines_and_words_bounding_boxes


