import pytesseract
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
from GeneralUtils.utils import *
import cv2
import numpy as np
from OcrPipeline.ocr_inference_pipeline import OcrInferencePipeline
from Data.SyntheticData.euclidean_utils import *


our_ocr = OcrInferencePipeline(
    'ocr_with_paragraphs',
    font_file=None,
    detection_name='detection_type_2_type_3_0.25',
    recognition_names=('recognizer2',),
    script_identification_name=None,
    load_pipeline_file=False,
    detection_batch_size=1,
    crop=None,
    detection_gpu_ids='0',
    detection_num_workers=1,
    detection_padding=1,
    detection_binary_mask_threshold=0.98,
    detection_post_processing=False,
    detection_post_processing_methods=('size_based_nms', 'area_threshold', 'length_threshold'),
    detection_post_processing_params={'intersection_thr': 0.5, 'area_threshold': 30, 'length_threshold': 5},
    every_word_preprocessing=(('maximize_contrast', None),),
    recognitions_gpu_ids=('0',),
    recognition_chunking_params={'ar_to_divide': 8},
    max_image_shape=None
)


def get_combined_pred(ocr_res):
    fp2ocr = {item['file']: item['prediction'] for item in ocr_res}
    for fp in tqdm(fp2ocr.keys()):
        fp2paragraphs[fp] = infer_paragraphs_image(fp)
    combined_res = {}
    for fp, paragraphs in tqdm(fp2paragraphs.items()):
        ocr = fp2ocr[fp]
        line_idx_used = []
        page_txt = []
        for paragraph in paragraphs:
            lines = []
            for iline, line in enumerate(ocr):
                if intersection_2D(line['line_bounding_box'], paragraph) >= (
                        0.8 * area_2D(line['line_bounding_box'])) and iline not in line_idx_used:
                    lines.append(line)
                    line_idx_used.append(iline)
            lines.sort(key=lambda l: l['line_bounding_box'][1])
            lines_x = []
            iline = 0
            while iline < len(lines):
                if iline == (len(lines) - 1):
                    lines_x.append(lines[iline])
                    break
                merged_lines = [lines[iline]]
                while (iline + 1) < len(lines) and vertical_1D_iou(merged_lines[-1]['line_bounding_box'],
                                                                   lines[iline + 1]['line_bounding_box']) >= 0.8:
                    merged_lines.append(lines[iline + 1])
                    iline += 1
                merged_lines.sort(key=lambda x: x['line_bounding_box'][0])
                lines_x.append({
                    'line_bounding_box': [
                        min([line['line_bounding_box'][0] for line in merged_lines]),
                        min([line['line_bounding_box'][1] for line in merged_lines]),
                        max([line['line_bounding_box'][2] for line in merged_lines]),
                        max([line['line_bounding_box'][3] for line in merged_lines])
                    ],
                    'line_text': ' '.join([line['line_text'] for line in merged_lines]),
                })
                iline += 1
            lines_x.sort(key=lambda l: l['line_bounding_box'][1])
            if len(lines_x) > 0:
                pargraph_txt = lines_x[0]['line_text']
                if len(lines_x) > 1:
                    for l in lines_x[1:]:
                        pargraph_txt += ' ' + l['line_text']
                        # if pargraph_txt[-1] == '-':
                        #     pargraph_txt = pargraph_txt + l['line_text']
                        # else:
                        #     pargraph_txt += ' ' + l['line_text']
                page_txt.append(pargraph_txt)
        for iline, line in enumerate(ocr):
            if iline not in line_idx_used:
                page_txt.append(line['line_text'])
        combined_res[fp] = '\n'.join(page_txt)
    combined_res = [{'file': k, 'prediction': v} for k, v in combined_res.items()]
    return combined_res


def infer_combined_folder(folder_path):
    fp2ocr = our_ocr.inference_whole_images(folder_path)
    return get_combined_pred(fp2ocr)


def infer_ours_combined_datasets(root_folder):
    root_dst_folder = os.path.join(root_folder, 'dst')
    for dataset in os.listdir(root_dst_folder):
        dataset_path = os.path.join(root_dst_folder, dataset)
        if os.path.isdir(dataset_path):
            for aug_type in os.listdir(dataset_path):
                aug_type_path = os.path.join(dataset_path, aug_type)
                if os.path.isdir(aug_type_path):
                    for num_augs in os.listdir(aug_type_path):
                        num_augs_path = os.path.join(aug_type_path, num_augs)
                        if os.path.isdir(num_augs_path):
                            infer_combined_folder(num_augs_path)


def infer_paragraphs_image(image_path):
    psm = 3
    oem = 1
    config = f'--psm {psm} --oem {oem}'
    image = Image.open(image_path)
    # Get full word-level data
    layout_data = pytesseract.image_to_data(image, config=config, lang='eng', output_type=pytesseract.Output.DICT)
    blocks = {}
    last_paragraph = None
    paragraphs = []
    for lidx, level in enumerate(layout_data['level']):
        block_num = layout_data['block_num'][lidx]
        par_num = layout_data['par_num'][lidx]
        if level == 2:  # block
            blocks[block_num] = {
                'bbox': [
                    layout_data['left'][lidx],
                    layout_data['top'][lidx],
                    layout_data['left'][lidx] + layout_data['width'][lidx],
                    layout_data['top'][lidx] + layout_data['height'][lidx]
                ],
                'paragraphs': {}
            }
        elif level == 3:
            blocks[block_num]['paragraphs'][par_num] = [
                layout_data['left'][lidx],
                layout_data['top'][lidx],
                layout_data['left'][lidx] + layout_data['width'][lidx],
                layout_data['top'][lidx] + layout_data['height'][lidx]
            ]
            last_paragraph = blocks[block_num]['paragraphs'][par_num]
        elif level == 5:
            if last_paragraph not in paragraphs and len(layout_data['text'][lidx].replace(' ', '')) > 0:
                paragraphs.append(last_paragraph)
    return paragraphs

