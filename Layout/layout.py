import numpy as np
from Data.SyntheticData.euclidean_utils import *
from tqdm import tqdm


def merge_dicts(groups_lines):
    return [
        [
            min([i['line_bounding_box'][0] for i in gl]),
            min([i['line_bounding_box'][1] for i in gl]),
            max([i['line_bounding_box'][2] for i in gl]),
            max([i['line_bounding_box'][3] for i in gl])
        ] for gl in groups_lines]


def merge_lists(groups_lines):
    return [
        [
            min([i[0] for i in gl]),
            min([i[1] for i in gl]),
            max([i[2] for i in gl]),
            max([i[3] for i in gl])
        ] for gl in groups_lines]


def merge_x_axis(group_lines):
    words = [i for l in group_lines for i in l['words']]
    words.sort(key=lambda x: x['bounding_box'][0])
    groups_words = []
    while len(words) > 0:
        for ig in range(len(groups_words)):
            g = groups_words[ig]
            ys_mean = np.mean([w['bounding_box'][1] for w in g])
            ye_mean = np.mean([w['bounding_box'][3] for w in g])
            if intersection_1D(ys_mean, ye_mean, words[0]['bounding_box'][1], words[0]['bounding_box'][3]) >= 0.5 * (words[0]['bounding_box'][3] - words[0]['bounding_box'][1]):
                groups_words[ig].append(words[0])
                break
        else:
            groups_words.append([words[0]])
        words.pop(0)
    new_lines = [{'line_bounding_box': merge_lists([[i['bounding_box'] for i in g]])[0], 'line_text': ' '.join([i['text'] for i in g]), 'words': g} for g in groups_words]
    return new_lines


def extract_layout(ocr_res, convert_to_text=True):
    res = {}
    for image_path, lines in tqdm(ocr_res.items()):
        xs = [[l['line_bounding_box'][0], l['line_bounding_box'][2]] for l in lines]
        groups_xs = []
        for sx, ex in xs:
            for ig in range(len(groups_xs)):
                if abs(np.mean([i[0] for i in groups_xs[ig]]) - sx) < 15 or abs(
                        np.mean([i[1] for i in groups_xs[ig]]) - ex) < 15:
                    groups_xs[ig].append([sx, ex])
                    break
            else:
                groups_xs.append([[sx, ex]])
        groups_xs.sort(key=lambda x: len(x), reverse=True)
        groups_lines = [[l for l in lines if [l['line_bounding_box'][0], l['line_bounding_box'][2]] in g] for g in
                        groups_xs]
        paragraphs = merge_dicts(groups_lines)
        paragraphs.sort(key=lambda x: area_2D(x), reverse=True)
        paragraphs_to_del_idxs = []
        for ip1 in range(len(paragraphs)):
            for ip2 in range(len(paragraphs)):
                if ip2 > ip1 and intersection_2D(paragraphs[ip1], paragraphs[ip2]) >= 0.8 * area_2D(paragraphs[ip2]):
                    paragraphs_to_del_idxs.append(ip2)
                    paragraphs[ip1] = merge_lists([[paragraphs[ip1], paragraphs[ip2]]])[0]
        paragraphs = [p for ip, p in enumerate(paragraphs) if ip not in paragraphs_to_del_idxs]
        paragraphs_lines = [
            [l for l in lines if intersection_2D(l['line_bounding_box'], p) >= 0.9 * area_2D(l['line_bounding_box'])]
            for p in paragraphs]
        header = []
        footer = []
        paragraphs_to_del_idxs = []
        ys = [l['line_bounding_box'][1] for l in lines]
        for ip, p in enumerate(paragraphs_lines):
            if len(p) == 1 and (p[0]['line_bounding_box'][1] - min(ys)) <= 15:
                header.append(p[0])
                paragraphs_to_del_idxs.append(ip)
            elif len(p) == 1 and (p[0]['line_bounding_box'][1] - max(ys)) <= 15:
                footer.append(p[0])
                paragraphs_to_del_idxs.append(ip)
        header = [header]
        footer = [footer]
        main_paragraphs = [p for ip, p in enumerate(paragraphs) if ip not in paragraphs_to_del_idxs]
        main_paragraphs_lines = [p for ip, p in enumerate(paragraphs_lines) if ip not in paragraphs_to_del_idxs]
        main_paragraphs_zipped = list(zip(main_paragraphs, main_paragraphs_lines))
        main_paragraphs_zipped.sort(key=lambda x: x[0][0])
        main_paragraphs_lines = [i[1] for i in main_paragraphs_zipped]
        main_paragraphs_lines_remove_margins = []
        if len(header[0]) > 0:
            bb = merge_dicts(header)[0]
            for g in main_paragraphs_lines:
                new_g = []
                for l in g:
                    if intersection_1D(l['line_bounding_box'][1], l['line_bounding_box'][3], bb[1], bb[3]) < 0.45 * (bb[3] - bb[1] + 1):
                        new_g.append(l)
                    else:
                        header[0].append(l)
                main_paragraphs_lines_remove_margins.append(new_g)
        else:
            main_paragraphs_lines_remove_margins = main_paragraphs_lines
        main_paragraphs_lines_remove_margins2 = []
        if len(footer[0]) > 0:
            bb = merge_dicts(footer)[0]
            for g in main_paragraphs_lines_remove_margins:
                new_g = []
                for l in g:
                    if intersection_1D(l['line_bounding_box'][1], l['line_bounding_box'][3], bb[1], bb[3]) < 0.45 * (bb[3] - bb[1] + 1):
                        new_g.append(l)
                    else:
                        footer[0].append(l)
                main_paragraphs_lines_remove_margins2.append(new_g)
        else:
            main_paragraphs_lines_remove_margins2 = main_paragraphs_lines_remove_margins
        main_paragraphs_lines = main_paragraphs_lines_remove_margins2
        main_paragraphs_lines = [g for g in main_paragraphs_lines if len(g) > 0]
        main_paragraphs_lines = [merge_x_axis(g) for g in main_paragraphs_lines]
        for ip in range(len(main_paragraphs_lines)):
            main_paragraphs_lines[ip].sort(key=lambda x: x['line_bounding_box'][1])
        header[0].sort(key=lambda x: x['line_bounding_box'][0])
        footer[0].sort(key=lambda x: x['line_bounding_box'][0])
        main_paragraphs_lines.sort(key=lambda x: min([l['line_bounding_box'][1] for l in x]))
        used_idxs = []
        vertical_groups = []
        main_paragraphs_lines_sorted = []
        for ip1, p1 in enumerate(main_paragraphs_lines):
            if ip1 not in used_idxs:
                vertical_groups.append([ip1])
                used_idxs.append(ip1)
                start1 = min([l['line_bounding_box'][1] for l in p1])
                end1 = max([l['line_bounding_box'][3] for l in p1])
                for ip2, p2 in enumerate(main_paragraphs_lines):
                    if ip2 > ip1:
                        start2 = min([l['line_bounding_box'][1] for l in p2])
                        end2 = max([l['line_bounding_box'][3] for l in p2])
                        if intersection_1D(start1, end1, start2, end2) > 0:
                            vertical_groups[-1].append(ip2)
                            used_idxs.append(ip2)
        for vertical_group in vertical_groups:
            group_ver = [main_paragraphs_lines[item] for item in vertical_group]
            group_ver.sort(key=lambda x: min([l['line_bounding_box'][0] for l in x]))
            main_paragraphs_lines_sorted += group_ver
        main_paragraphs_lines = main_paragraphs_lines_sorted + footer + header
        main_paragraphs_lines = [g for g in main_paragraphs_lines if len(g) > 0]
        main_paragraphs_lines = [
            {
                'line_bounding_box': merge_dicts([g])[0],
                'line_text': '\n'.join([l['line_text'] for l in g]),
                'words': [w for l in g for w in l['words']]
            } for g in main_paragraphs_lines
        ]
        if not convert_to_text:
            res[image_path] = main_paragraphs_lines  #ocr_res[image_path]  #main_paragraphs_lines
        else:
            res[image_path] = '\n'.join([g['line_text'] for g in main_paragraphs_lines])
    return res


