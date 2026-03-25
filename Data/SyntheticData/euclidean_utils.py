def union_1D(start1, end1, start2, end2):
    if start1 <= start2 <= end1:
        return max(end1, end2) - start1 + 1
    elif start2 <= start1 <= end2:
        return max(end1, end2) - start2 + 1
    return end1 - start1 + end2 - start2 + 2


def intersection_1D(start1, end1, start2, end2):
    if start1 <= start2 <= end1:
        return min(end1, end2) - start2 + 1
    elif start2 <= start1 <= end2:
        return min(end1, end2) - start1 + 1
    return 0


def bounding_box_list_to_dict(bb):
    if type(bb) == dict:
        return bb
    bb = {
        'left': bb[0],
        'top': bb[1],
        'right': bb[2],
        'bottom': bb[3]
    }
    return bb


def bounding_box_dict_to_list(bb):
    if type(bb) in [list, tuple]:
        return bb
    bb = [bb['left'], bb['top'], bb['right'], bb['bottom']]
    return bb


def vertical_1D_iou(bb1, bb2):
    bb1 = bounding_box_list_to_dict(bb1)
    bb2 = bounding_box_list_to_dict(bb2)
    bb1_y_start, bb1_y_end = bb1['top'], bb1['bottom']
    bb2_y_start, bb2_y_end = bb2['top'], bb2['bottom']
    union = float(union_1D(bb1_y_start, bb1_y_end, bb2_y_start, bb2_y_end))
    if union > 0:
        iou = float(intersection_1D(bb1_y_start, bb1_y_end, bb2_y_start, bb2_y_end)) / union
    else:
        iou = 0.0
    return iou


def horizontal_1D_iou(bb1, bb2):
    bb1 = bounding_box_list_to_dict(bb1)
    bb2 = bounding_box_list_to_dict(bb2)
    bb1_x_start, bb1_x_end = bb1['left'], bb1['right']
    bb2_x_start, bb2_x_end = bb2['left'], bb2['right']
    union = float(union_1D(bb1_x_start, bb1_x_end, bb2_x_start, bb2_x_end))
    if union > 0:
        iou = float(intersection_1D(bb1_x_start, bb1_x_end, bb2_x_start, bb2_x_end)) / union
    else:
        iou = 0.0
    return iou


def area_2D(bb):
    bb = bounding_box_list_to_dict(bb)
    area = max(0, bb['bottom'] - bb['top'] + 1) * max(0, bb['right'] - bb['left'] + 1)
    return area


def intersection_2D(bb1, bb2):
    bb1 = bounding_box_list_to_dict(bb1)
    bb2 = bounding_box_list_to_dict(bb2)
    int_x1 = max(bb1['left'], bb2['left'])
    int_y1 = max(bb1['top'], bb2['top'])
    int_x2 = min(bb1['right'], bb2['right'])
    int_y2 = min(bb1['bottom'], bb2['bottom'])
    int_area = area_2D([int_y1, int_x1, int_y2, int_x2])
    return int_area


def iou_2D(bb1, bb2):
    bb1 = bounding_box_list_to_dict(bb1)
    bb2 = bounding_box_list_to_dict(bb2)
    bb1_area = area_2D(bb1)
    bb2_area = area_2D(bb2)
    int_area = intersection_2D(bb1, bb2)
    denominator = float(bb1_area + bb2_area - int_area)
    if denominator == 0.0:
        return 0.0
    return int_area / denominator


def bounding_box_minus_inner_bounding_box(bb1, bb2):
    bb1 = bounding_box_list_to_dict(bb1)
    bb2 = bounding_box_list_to_dict(bb2)
    sub_x1 = bb1['left'] if bb2['left'] > bb1['left'] else (bb2['right'] + 1)
    sub_y1 = bb1['top'] if bb2['top'] > bb1['top'] else (bb2['bottom'] + 1)
    sub_x2 = bb1['right'] if (sub_x1 == (bb2['right'] + 1)) else (bb2['left'] - 1)
    sub_y2 = bb1['bottom'] if (sub_y1 == (bb2['bottom'] + 1)) else (bb2['top'] - 1)
    return {
        'left': sub_x1,
        'top': sub_y1,
        'right': sub_x2,
        'bottom': sub_y2
    }

