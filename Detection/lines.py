import numpy as np
from skimage.measure import label, regionprops
import cv2
from Data.SyntheticData.euclidean_utils import *


def lines_core(segmentation_output, binary_mask_lines_threshold=0.9):
    preds2 = (segmentation_output >= binary_mask_lines_threshold) * 1
    preds2 = cv2.blur(preds2.astype(np.float32), (21, 1))
    #    preds2 = cv2.blur(src=preds2.astype(np.uint8), ksize=(21, 1))
    preds2 = (preds2 > 0.01) * 1
    bounding_boxes, connected_components_images = connected_components_filled(preds2, True)
    return bounding_boxes, connected_components_images


def connected_components_filled(preds, take_only_biggest=False):
    bounding_boxes, connected_components_images = connected_components(preds)
    bounding_boxes_and_images_filled = [(bounding_boxes[i], connected_components_images[i]) for i in range(len(bounding_boxes))]
    filled_mask = np.zeros(preds.shape)
    bounding_boxes_and_images_filled.sort(key=lambda x: area_2D(x[0]), reverse=True)
    for i in range(len(bounding_boxes_and_images_filled)):
        image_mask = bounding_boxes_and_images_filled[i][1]
        if take_only_biggest:
            cropped_mask = filled_mask[bounding_boxes_and_images_filled[i][0][1]: bounding_boxes_and_images_filled[i][0][3] + 1,
                           bounding_boxes_and_images_filled[i][0][0]: bounding_boxes_and_images_filled[i][0][2] + 1]
            image_mask = image_mask * (1 - (cropped_mask > 0) * 1)
        filled_mask[bounding_boxes_and_images_filled[i][0][1]: bounding_boxes_and_images_filled[i][0][3] + 1,
                           bounding_boxes_and_images_filled[i][0][0]: bounding_boxes_and_images_filled[i][0][2] + 1] += image_mask
    filled_mask = (filled_mask > 0) * 1
    bounding_boxes, connected_components_images = connected_components(filled_mask)
    return bounding_boxes, connected_components_images


def connected_components(preds):
    cc = label(preds)
    rp = regionprops(cc)
    bounding_boxes_and_images_filled = [([
        i.bbox[1],
        i.bbox[0],
        i.bbox[3] - 1,
        i.bbox[2] - 1
    ], i.image_convex * 1) for i in rp if i.bbox[0] != i.bbox[2] and i.bbox[1] != i.bbox[3]]
    connected_components_images = [i[1] for i in bounding_boxes_and_images_filled]
    bounding_boxes = [i[0] for i in bounding_boxes_and_images_filled]
    return bounding_boxes, connected_components_images


def line_segment_to_words(line_segmentation_output_np_at_mask, binary_mask_words_threshold):
    binary_mask_line = (line_segmentation_output_np_at_mask > binary_mask_words_threshold) * 1
    bounding_boxes_words, connected_components_word_images = connected_components_filled(binary_mask_line, True)
    return bounding_boxes_words, connected_components_word_images


def divide_heirarcical_line_segment_to_words(line_segmentation_output_np_at_mask, h, w, binary_mask_lines_threshold,
                                             binary_mask_words_threshold, padding=1, former_h_non_normal=None):
    bounding_boxes_words, connected_components_word_images = line_segment_to_words(line_segmentation_output_np_at_mask, binary_mask_words_threshold)
    heights = [(ibb, bb[3] - bb[1] + 1, connected_components_word_images[ibb]) for ibb, bb in enumerate(bounding_boxes_words)]
    heights_median = np.median([i[1] for i in heights])
    h_non_normal = 1.75 * heights_median
    if former_h_non_normal is not None:
        h_non_normal = min(h_non_normal, former_h_non_normal)
    new_bounding_boxes_words, new_connected_components_word_images = [], []
    for ibb, h, word_image in heights:
        if h > h_non_normal and binary_mask_words_threshold < 0.99:
            word_bb = bounding_boxes_words[ibb]
            word_segmentation_output_np_at_mask = line_segmentation_output_np_at_mask[word_bb[1]: word_bb[3] + 1, word_bb[0]: word_bb[2] + 1]
            partial_bounding_boxes_words = divide_heirarcical_line_segment_to_words(word_segmentation_output_np_at_mask * word_image, h, w, binary_mask_lines_threshold, 0.99,
                                                                                    padding, h_non_normal)
            partial_bounding_boxes_words = [[
                i[0] + word_bb[0],
                i[1] + word_bb[1],
                i[2] + word_bb[0],
                i[3] + word_bb[1]
            ] for i in partial_bounding_boxes_words]
            new_bounding_boxes_words += partial_bounding_boxes_words
        else:
            new_bounding_boxes_words.append(bounding_boxes_words[ibb])
            new_connected_components_word_images.append(word_image)
    return new_bounding_boxes_words


def find_lines(segmentation_output, h, w, binary_mask_lines_threshold,
                                             binary_mask_words_threshold, padding=1, former_h_non_normal=None):
    bounding_boxes_lines, connected_components_lines_images = lines_core(segmentation_output, binary_mask_lines_threshold)
    lines_and_words_bbs = []
    for idx_bb_line, bb_line in enumerate(bounding_boxes_lines):
        line_segmentation_output = segmentation_output[bb_line[1]: bb_line[3] + 1, bb_line[0]: bb_line[2] + 1]
        line_segmentation_output_at_mask = line_segmentation_output * connected_components_lines_images[idx_bb_line]
        bounding_boxes_words = divide_heirarcical_line_segment_to_words(line_segmentation_output_at_mask, h, w, binary_mask_lines_threshold, binary_mask_words_threshold,
                                                                        padding, former_h_non_normal)
        bounding_boxes_words.sort(key=lambda x: area_2D(x), reverse=True)
        line_and_words = {'line_bounding_box': bb_line, 'words': []}
        for ibb_word, bb_word in enumerate(bounding_boxes_words):
            bb_word_rel = [
                max(bb_word[0] + bb_line[0] - padding, 0),
                max(bb_word[1] + bb_line[1] - padding, 0),
                min(bb_word[2] + bb_line[0] + padding, w - 1),
                min(bb_word[3] + bb_line[1] + padding, h - 1)
            ]
            line_and_words['words'].append(bb_word_rel)
        line_and_words['words'].sort(key=lambda x: x[0])
        lines_and_words_bbs.append(line_and_words)
    return lines_and_words_bbs, bounding_boxes_lines


def split_lines(lines_and_words_bbs):
    lines_and_words_bbs_splitted = []
    for idx_line, line in enumerate(lines_and_words_bbs):
        bounding_boxes_words_splitted_into_sublines = []
        for ibb_word, bb_word_rel in enumerate(line['words']):
            found_subline = False
            for isubline, subline in enumerate(bounding_boxes_words_splitted_into_sublines):
                subline_words = subline['words']
                for word in subline_words:
                    if vertical_1D_iou(word, bb_word_rel) > 0.7:
                        bounding_boxes_words_splitted_into_sublines[isubline]['words'].append(bb_word_rel)
                        found_subline = True
                        break
            if not found_subline:
                bounding_boxes_words_splitted_into_sublines.append({'line_bounding_box': [], 'words': [bb_word_rel]})
        for isubline in range(len(bounding_boxes_words_splitted_into_sublines)):
            bounding_boxes_words_splitted_into_sublines[isubline]['line_bounding_box'] = [
                min([i[0] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']]),
                min([i[1] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']]),
                max([i[2] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']]),
                max([i[3] for i in bounding_boxes_words_splitted_into_sublines[isubline]['words']])
            ]
        lines_and_words_bbs_splitted += bounding_boxes_words_splitted_into_sublines
    return lines_and_words_bbs_splitted


def filter_lines_and_words(lines_and_words_bbs):
    new_lines_and_words_bbs = []
    lines_and_words_bbs.sort(key=lambda x: area_2D(x['line_bounding_box']), reverse=True)
    for il in range(len(lines_and_words_bbs)):
        l = lines_and_words_bbs[il]
        for il2 in range(len(new_lines_and_words_bbs)):
            l2 = new_lines_and_words_bbs[il2]
            if intersection_2D(l['line_bounding_box'], l2['line_bounding_box']) > 0.5 * area_2D(l['line_bounding_box']):
                new_lines_and_words_bbs[il2]['words'] += l['words']
                break
        else:
            new_lines_and_words_bbs.append(l)
    for il in range(len(new_lines_and_words_bbs)):
        new_lines_and_words_bbs[il]['line_bounding_box'] = [
            min([i[0] for i in new_lines_and_words_bbs[il]['words']]),
            min([i[1] for i in new_lines_and_words_bbs[il]['words']]),
            max([i[2] for i in new_lines_and_words_bbs[il]['words']]),
            max([i[3] for i in new_lines_and_words_bbs[il]['words']])
        ]
        new_lines_and_words_bbs[il]['words'].sort(key=lambda x: x[0])
        words = []
        for w in new_lines_and_words_bbs[il]['words']:
            for w2 in words:
                if intersection_2D(w, w2['bounding_box']) >= 0.8 * area_2D(w):
                    break
            else:
                words.append({'bounding_box': w})
        new_lines_and_words_bbs[il]['words'] = words
    return new_lines_and_words_bbs


def lines_and_words_extraction(segmentation_output, h, w, binary_mask_lines_threshold,
                                             binary_mask_words_threshold, padding=1):
    seg = segmentation_output.detach().cpu()[0, 1, :, :].numpy()
    line_and_words_bbs, _ = find_lines(seg, h, w, binary_mask_lines_threshold, binary_mask_words_threshold, padding)
    line_and_words_bbs_splitted = split_lines(line_and_words_bbs)
    line_and_words_bbs_filtered = filter_lines_and_words(line_and_words_bbs_splitted)
    return line_and_words_bbs_filtered

