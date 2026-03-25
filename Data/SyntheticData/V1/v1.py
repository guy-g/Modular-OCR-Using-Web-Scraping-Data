import os
import sys
import shutil
from uuid import uuid4
from PIL import Image, ImageDraw, ImageFont
import time
import json
import io
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import numpy as np
from skimage.measure import label
import base64
from tqdm import tqdm
from collections import Counter
import requests
import timeout_decorator


def save_to_json(dictionary, file_path):
    json_object = json.dumps(dictionary, indent=4)
    with open(file_path, mode='w', encoding='utf-8') as data_file_writer:
        data_file_writer.write(json_object)


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
                    "bbox": [bb['left'], bb['top'], bb['right'] - bb['left'], bb['bottom'] - bb['top']],
                    "text": None
                } for ibb, bb in enumerate(detection_data['tags_to_bounding_boxes_images'].values())
            ]
            coco_format_whole_dataset['categories'] = coco_format["categories"]
            coco_format_whole_dataset['annotations'] += coco_format["annotations"]
            coco_format_whole_dataset['images'] += coco_format["images"]
            coco_format_whole_dataset['images'][-1]["file_name"] = os.path.join(p, 'image.' + ext)
            coco_object = json.dumps(coco_format, indent=4)
            with open(os.path.join(p, 'detection_data_coco.json'), mode='w', encoding='utf-8') as data_file:
                data_file.write(coco_object)
    coco_format_whole_dataset_object = json.dumps(coco_format_whole_dataset, indent=4)
    with open(os.path.join(folder_path, 'detection_dataset_coco.json'), mode='w', encoding='utf-8') as data_file:
        data_file.write(coco_format_whole_dataset_object)


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
                        "bbox": [paragraph['left'], paragraph['top'], paragraph['right'] - paragraph['left'], paragraph['bottom'] - paragraph['top']]
                    } for ibb, (paragraph, words) in enumerate(layout_data['paragraphs'])
                ]
            }
            coco_format_whole_dataset['categories'] = coco_format["categories"]
            coco_format_whole_dataset['annotations'] += coco_format["annotations"]
            coco_format_whole_dataset['images'] += coco_format["images"]
            coco_format_whole_dataset['images'][-1]["file_name"] = os.path.join(p, 'image.' + ext)
            coco_object = json.dumps(coco_format, indent=4)
            with open(os.path.join(p, 'layout_data_coco.json'), mode='w', encoding='utf-8') as data_file:
                data_file.write(coco_object)
    coco_format_whole_dataset_object = json.dumps(coco_format_whole_dataset, indent=4)
    with open(os.path.join(folder_path, 'layout_dataset_coco.json'), mode='w', encoding='utf-8') as data_file:
        data_file.write(coco_format_whole_dataset_object)


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


COLOR_HEX_RANGE = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
TEXT_DECORATION_OPTIONS = ["none", "overline", "line-through", "underline", "underline overline"]
FONT_STYLE_OPTIONS = ["normal", "italic", "oblique"]
FONT_VARIANT_OPTIONS = ["normal", "small-caps"]
FONT_WEIGHT_OPTIONS = ["normal", "bold", "bolder", "lighter", 100, 200, 300, 400, 500, 600, 700, 800, 900]
FONT_FAMILY_OPTIONS = ["serif", "san-serif", "monospace", "cursive", "emoji", "math"]
FONT_STRETCH_OPTIONS = ["ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "normal", "semi-expanded", "expanded", "extra-expanded", "ultra-expanded"]
FONT_SIZE_RANGE = [5, 45]
LETTER_SPACING_RANGE = [1, 15]


MAX_WORDS_PER_PAGE_FOR_REPLACEMENT = 500
SELF_PARAGRAPHS_TAGS = {
    0: ['li', 'td', 'tr', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'pre', 'dt', 'dd', 'p'],
    1: ['p', 'dl']
}

CHANGE_BACKGROUND_JS_CODE = '''
const colors_range = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'];
function get_random_color() {{
    var random_color = '#';
    for (let ch = 0; ch < 6; ch++) {{
        random_color += colors_range[(Math.floor(Math.min(Math.random(), 0.999999) * colors_range.length))];
    }}
    return random_color;
}}
function change_background(element, color) {{
    for (var i=0; i < element.childNodes.length; i++) {{
        change_background(element.childNodes[i], color);
    }}
    if (element.nodeType === Node.ELEMENT_NODE) {{
        element.style.backgroundColor = color;
    }}
}}
var rand_color = get_random_color();
change_background(document.body, rand_color);
'''

MAKE_ELEMENT_INVISIBLE_BY_ID = '''
var word = document.getElementById(arguments[0]); 
word.style.webkitTextFillColor = 'rgb(255, 0, 0, 0)'; 
word.style.visibility = 'hidden';
return word.getBoundingClientRect();
'''

MAKE_ELEMENT_VISIBLE_BY_ID = '''
var word = document.getElementById(arguments[0]); 
word.style = ''; 
return true;
'''

MAKE_ELEMENT_INVISIBLE = '''
arguments[0].style.visibility = 'hidden';
return arguments[0].getBoundingClientRect();
'''

MAKE_ELEMENT_VISIBLE = '''
arguments[0].style = ''; 
return true;
'''

HIDE_OPEN_SHADOW_ROOTS = '''
function hide_open_shadow_roots(node) {{
    for (var i=0; i < node.childNodes.length; i++) {{
        if (node.childNodes[i].nodeType === Node.ELEMENT_NODE && node.childNodes[i].shadowRoot != null) {{
            node.childNodes[i].style.visibility = 'hidden';
            node.childNodes[i].remove();
        }} else if (node.childNodes[i].nodeType === Node.ELEMENT_NODE) {{
            hide_open_shadow_roots(node.childNodes[i]);
        }}
    }}
}}
hide_open_shadow_roots(document);
return true;
'''

HIDE_ICONS = '''
function hide_open_shadow_roots(node) {{
    for (var i=0; i < node.childNodes.length; i++) {{
        if (node.childNodes[i].nodeType === Node.ELEMENT_NODE && node.childNodes[i].shadowRoot != null) {{
            node.childNodes[i].style.visibility = 'hidden';
            node.childNodes[i].remove();
        }} else if (node.childNodes[i].nodeType === Node.ELEMENT_NODE) {{
            hide_open_shadow_roots(node.childNodes[i]);
        }}
    }}
}}
hide_open_shadow_roots(document);
return true;
'''


def EXTRACT_DATA_JS_CODE(probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                         probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction='ltr', enable_icon_or_emoji_text=False, dictionary=None, probability_replace_word_from_dictionary=0.0):
    direction = ' direction: {};'.format(force_direction) if force_direction is not None else ''
    enable_icon_or_emoji_text = 'true' if enable_icon_or_emoji_text else 'false'
    if dictionary is not None and probability_replace_word_from_dictionary > 0.0:
        random_words = [dictionary.get_word() for _ in range(MAX_WORDS_PER_PAGE_FOR_REPLACEMENT)]
    else:
        random_words = []
    js_code = '''
const enable_icon_or_emoji_text = {};
const random_words = {};
const probability_replace_word_from_dictionary = {};
var word_ids = [];
var word_contents = [];
var word_bounding_boxes = [];
var image_elements = [];
var image_elements_final = [];
var image_bounding_boxes = [];
var saved_text_nodes_and_parents = [];
const forbidden_tags = ['script', 'noscript', 'style', 'title', 'head', 'iframe'];
const tags_to_remove_inner_text = ['option', 'input', 'textarea'];
const image_tags = ['img', 'svg', 'video', 'embed', 'canvas'];
const non_taking_characters = [' ', '    ', '\\u00A0', '\\u0009', '\\n', '\\t', '\\u2003', '\\u2002'];
if (enable_icon_or_emoji_text) {{
    var icon_classes = [];
}} else {{
    var icon_classes = ['material-icons', 'fa', 'fas', 'glyphicon', 'fab'];
}}
const remove_psuedo_elements_code = "*::after {{content: none !important;}}*::before {{content: none !important;}}*::marker {{content: none !important;}}*:after {{content: none !important;}}*:before {{content: none !important;}}*:marker {{content: none !important;}}"
const colors_range = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'];
const TEXT_DECORATION_OPTIONS = ["none", "overline", "line-through", "underline", "underline overline"];
const FONT_STYLE_OPTIONS = ["normal", "italic", "oblique"];
const FONT_VARIANT_OPTIONS = ["normal", "small-caps"];
const FONT_WEIGHT_OPTIONS = ["normal", "bold", "bolder", "lighter", 100, 200, 300, 400, 500, 600, 700, 800, 900];
const FONT_FAMILY_OPTIONS = ["serif", "san-serif", "monospace", "cursive", "emoji", "math"];
const FONT_STRETCH_OPTIONS = ["ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "normal", "semi-expanded", "expanded", "extra-expanded", "ultra-expanded"];
const FONT_SIZE_RANGE = [5, 45];
const probability_change_text_color = {};
const probability_change_font = {};
const probability_change_font_size = {};
const probability_change_font_decoration = {};
const probability_change_font_style = {};
const probability_change_font_variant = {};
const probability_change_font_weight = {};
const probability_change_font_stretch = {};
var random_words_idx = 0;
var layout_data = {{}};
var words_layout = [];
const SELF_PARAGRAPHS_TAGS = {};

function get_random_color() {{
    var random_color = '#';
    for (let ch = 0; ch < 6; ch++) {{
        random_color += colors_range[(Math.floor(Math.min(Math.random(), 0.999999) * colors_range.length))];
    }}
    return random_color;
}}

function uuidv4() {{
  return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
  );
}}

function remove_psuedo_elements() {{
    var styleSheet = document.createElement("style");
    styleSheet.innerText = remove_psuedo_elements_code;
    document.head.appendChild(styleSheet);
}}

function find_text_nodes(node) {{
    for (var i=0; i < node.childNodes.length; i++) {{
        if (node.childNodes[i].nodeType === Node.TEXT_NODE && node.childNodes[i].wholeText.length > 0 && (!(node.nodeType === Node.ELEMENT_NODE) || (!forbidden_tags.includes(node.tagName.toLowerCase())))) {{
            saved_text_nodes_and_parents.push([node.childNodes[i], node]);
        }} else if (node.childNodes[i].shadowRoot != null) {{
            node.childNodes[i].style.visibility = 'hidden';
            node.childNodes[i].remove();
        }} else if (node.childNodes[i].nodeType === Node.ELEMENT_NODE && !forbidden_tags.includes(node.childNodes[i].tagName.toLowerCase()) && getComputedStyle(node.childNodes[i]).visibility == 'visible') {{
            if (node.childNodes[i].value) {{
                node.childNodes[i].value = '';
            }}
            if (node.childNodes[i].alt) {{
                node.childNodes[i].alt = '';
            }}
            if (node.childNodes[i].placeholder) {{
                node.childNodes[i].placeholder = '';
            }}
            if (node.childNodes[i].src) {{
                node.childNodes[i].src = '';
            }}
            if (tags_to_remove_inner_text.includes(node.childNodes[i].tagName.toLowerCase())) {{
                node.childNodes[i].innerText = '';
            }}
            find_text_nodes(node.childNodes[i]);
        }} else if (node.childNodes[i].nodeType === Node.ELEMENT_NODE && node.childNodes[i].tagName.toLowerCase() == 'iframe' && getComputedStyle(node.childNodes[i]).visibility == 'visible') {{
            node.childNodes[i].style.visibility = 'hidden';
        }}
        if (node.childNodes[i].nodeType === Node.ELEMENT_NODE && (image_tags.includes(node.childNodes[i].tagName.toLowerCase()) || getComputedStyle(node.childNodes[i]).backgroundImage != "none") && getComputedStyle(node.childNodes[i]).visibility == 'visible') {{
            var is_only_image = image_tags.includes(node.childNodes[i].tagName.toLowerCase());
            if (node.childNodes[i].alt) {{
                node.childNodes[i].alt = '';
            }}
            image_elements.push([node.childNodes[i], is_only_image]);
        }}
    }}
}}


function find_closest_paragraph(elem, source_elem, level=0) {{
    if (SELF_PARAGRAPHS_TAGS[level].includes(elem.tagName.toLowerCase())) {{
        return elem;
    }} else if (elem.parentElement) {{
        return find_closest_paragraph(elem.parentElement, source_elem, level);
    }}
    if (SELF_PARAGRAPHS_TAGS.hasOwnProperty(level + 1)) {{
        return find_closest_paragraph(source_elem, source_elem, level + 1);
    }} else {{
        return source_elem;
    }}
}}


function change_text_nodes() {{
    var wrapping_words = [];
    for (var k=0; k < saved_text_nodes_and_parents.length; k++) {{
        var [text_node, parent_node] = saved_text_nodes_and_parents[k];

        var intext_words = [];
        var latest_index_taking_characters = 0;
        var latest_index_non_taking_characters = 0;

        for (var ch=0; ch < text_node.wholeText.length; ch++) {{
            if (non_taking_characters.includes(text_node.wholeText[ch].toLowerCase())) {{
                if (latest_index_taking_characters < ch) {{
                    var word = text_node.wholeText.substring(latest_index_taking_characters, ch);
                    intext_words.push(['word', word]);
                }}
                if (ch == (text_node.wholeText.length - 1)) {{ 
                    var non_word = text_node.wholeText.substring(latest_index_non_taking_characters);
                    intext_words.push(['non_word', non_word]);
                }}
                latest_index_taking_characters = ch + 1;
            }} else {{
                if (latest_index_non_taking_characters < ch) {{
                    var non_word = text_node.wholeText.substring(latest_index_non_taking_characters, ch);
                    intext_words.push(['non_word', non_word]);
                }}
                if (ch == (text_node.wholeText.length - 1)) {{ 
                    var word = text_node.wholeText.substring(latest_index_taking_characters);
                    intext_words.push(['word', word]);
                }}
                latest_index_non_taking_characters = ch + 1;
            }}
        }}

        var wrapping_span = document.createElement("span");
        wrapping_span.style = "unicode-bidi: bidi-override;{}";
        for (var i=0; i < intext_words.length; i++) {{
            var [text_type, text_content] = intext_words[i];
            if (text_type == 'word') {{
                if (Math.random() < probability_replace_word_from_dictionary && random_words.length > 0) {{
                    text_content = random_words[random_words_idx % random_words.length];
                    random_words_idx = random_words_idx + 1;
                }}
                var new_span = document.createElement("span");
                new_span.id = uuidv4();
                new_span.style = "white-space: nowrap; unicode-bidi: bidi-override;{}";
                var word_text = document.createTextNode(text_content);
                new_span.appendChild(word_text);
                wrapping_span.appendChild(new_span);
                wrapping_words.push([new_span.id, text_content]);
                var parent_paragraph = find_closest_paragraph(parent_node, parent_node);
                layout_data[new_span.id] = [k, parent_node.tagName, parent_paragraph, parent_paragraph.tagName];
            }} else {{
                var new_span = document.createElement("span");
                var word_text = document.createTextNode(text_content);
                new_span.appendChild(word_text)
                wrapping_span.appendChild(new_span);
            }}
        }}
        if (Math.random() < probability_change_text_color) {{
            wrapping_span.style.webkitTextFillColor = get_random_color();
        }}
        //if (Math.random() < probability_change_font_decoration) {{
        //    wrapping_span.style.textDecoration = TEXT_DECORATION_OPTIONS[(Math.floor(Math.min(Math.random(), 0.999999) * TEXT_DECORATION_OPTIONS.length))];
        //}}
        if (Math.random() < probability_change_font_style) {{
            wrapping_span.style.fontStyle = FONT_STYLE_OPTIONS[(Math.floor(Math.min(Math.random(), 0.999999) * FONT_STYLE_OPTIONS.length))];
        }}
        if (Math.random() < probability_change_font_variant) {{
            wrapping_span.style.fontVariant = FONT_VARIANT_OPTIONS[(Math.floor(Math.min(Math.random(), 0.999999) * FONT_VARIANT_OPTIONS.length))];
        }}
        if (Math.random() < probability_change_font_weight) {{
            wrapping_span.style.fontWeight = FONT_WEIGHT_OPTIONS[(Math.floor(Math.min(Math.random(), 0.999999) * FONT_WEIGHT_OPTIONS.length))];
        }}
        if (Math.random() < probability_change_font_stretch) {{
            wrapping_span.style.fontStretch = FONT_STRETCH_OPTIONS[(Math.floor(Math.min(Math.random(), 0.999999) * FONT_STRETCH_OPTIONS.length))];
        }}
        if (Math.random() < probability_change_font_size) {{
            wrapping_span.style.fontSize = String(Math.floor(Math.random() * FONT_SIZE_RANGE[1]) + FONT_SIZE_RANGE[0]) + "px";
        }}
        parent_node.replaceChild(wrapping_span, text_node);
    }}
    for (var wrapping_idx=0; wrapping_idx < wrapping_words.length; wrapping_idx++) {{
        var [span_id, span_content] = wrapping_words[wrapping_idx];
        var bounding_box = document.getElementById(span_id).getBoundingClientRect();        
        var centery = Math.floor((bounding_box.top + bounding_box.bottom) / 2);
        var centerx = Math.floor((bounding_box.left + bounding_box.right) / 2);
        var centerx_half_left = Math.floor((bounding_box.left + centerx) / 2);
        var centerx_half_right = Math.floor((bounding_box.right + centerx) / 2);
        var centery_half_top = Math.floor((bounding_box.top + centery) / 2);
        var centery_half_bottom = Math.floor((bounding_box.bottom + centery) / 2);
        var points_in_front = (document.elementFromPoint(centerx, centery) == document.getElementById(span_id)) * 1 +
                                (document.elementFromPoint(centerx_half_left, centery) == document.getElementById(span_id)) * 1 +
                                (document.elementFromPoint(centerx_half_right, centery) == document.getElementById(span_id)) * 1 +
                                (document.elementFromPoint(centerx, centery_half_top) == document.getElementById(span_id)) * 1 +
                                (document.elementFromPoint(centerx, centery_half_bottom) == document.getElementById(span_id)) * 1;
        var may_contain_icon = false;
        for (var cl=0; cl < document.getElementById(span_id).parentElement.parentElement.classList.length; cl++) {{
            if (!enable_icon_or_emoji_text) {{
                if (/\p{{Extended_Pictographic}}/u.test(span_content) || icon_classes.includes(document.getElementById(span_id).parentElement.parentElement.classList[cl].toLowerCase())) {{
                    may_contain_icon = true;
                }}
            }}
        }}
        if ((bounding_box.top >= 0 && bounding_box.top < window.innerHeight) &&
            (bounding_box.top < bounding_box.bottom) &&
            (bounding_box.bottom >= 0 && bounding_box.bottom < window.innerHeight) &&
            (bounding_box.left >= 0 && bounding_box.left < window.innerWidth) &&
            (bounding_box.left < bounding_box.right) &&
            (bounding_box.right >= 0 && bounding_box.right < window.innerWidth) &&
            points_in_front >= 5 && (!may_contain_icon)) {{
            word_ids.push(span_id);
            word_contents.push(span_content);
            word_bounding_boxes.push(bounding_box);
            words_layout.push(layout_data[span_id]);
        }} else {{
            document.getElementById(span_id).style.webkitTextFillColor = 'rgb(255, 0, 0, 0)';
            document.getElementById(span_id).style.visibility = 'hidden';
        }}
    }}
}}

function get_images_locations() {{
    for (var image_idx=0; image_idx < image_elements.length; image_idx++) {{
        var bounding_box = image_elements[image_idx][0].getBoundingClientRect();
        var centery = Math.floor((bounding_box.top + bounding_box.bottom) / 2);
        var centerx = Math.floor((bounding_box.left + bounding_box.right) / 2);
        var centerx_half_left = Math.floor((bounding_box.left + centerx) / 2);
        var centerx_half_right = Math.floor((bounding_box.right + centerx) / 2);
        var centery_half_top = Math.floor((bounding_box.top + centery) / 2);
        var centery_half_bottom = Math.floor((bounding_box.bottom + centery) / 2);
        var points_in_front = (document.elementFromPoint(centerx, centery) == image_elements[image_idx][0]) * 1 +
                                (document.elementFromPoint(centerx_half_left, centery) == image_elements[image_idx][0]) * 1 +
                                (document.elementFromPoint(centerx_half_right, centery) == image_elements[image_idx][0]) * 1 +
                                (document.elementFromPoint(centerx, centery_half_top) == image_elements[image_idx][0]) * 1 +
                                (document.elementFromPoint(centerx, centery_half_bottom) == image_elements[image_idx][0]) * 1;
        var in_screen = (bounding_box.top >= 0 && bounding_box.top <= window.innerHeight) &&
            (bounding_box.top < bounding_box.bottom) &&
            (bounding_box.bottom >= 0 && bounding_box.bottom <= window.innerHeight) &&
            (bounding_box.left >= 0 && bounding_box.left <= window.innerWidth) &&
            (bounding_box.left < bounding_box.right) &&
            (bounding_box.right >= 0 && bounding_box.right <= window.innerWidth);
        var part_in_screen = ((bounding_box.top >= 0 && bounding_box.top <= window.innerHeight) || (bounding_box.bottom >= 0 && bounding_box.bottom <= window.innerHeight)) &&
            (bounding_box.top < bounding_box.bottom) &&
            ((bounding_box.left >= 0 && bounding_box.left <= window.innerWidth) || (bounding_box.right >= 0 && bounding_box.right <= window.innerWidth)) &&
            (bounding_box.left < bounding_box.right);
        if (part_in_screen && points_in_front >= 0) {{
            image_bounding_boxes.push(bounding_box);
            image_elements_final.push(image_elements[image_idx][0]);
        }} else if (image_elements[image_idx][1]) {{
            image_elements[image_idx][0].style.visibility = 'hidden';
        }}
    }}
}}
remove_psuedo_elements();
find_text_nodes(document);
change_text_nodes();
get_images_locations();
return [word_ids, word_contents, word_bounding_boxes, image_bounding_boxes, words_layout, image_elements_final];
    '''.format(enable_icon_or_emoji_text, random_words, probability_replace_word_from_dictionary, probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
               probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, SELF_PARAGRAPHS_TAGS, direction, direction)
    return js_code


def CHANGE_FONT_JS_CODE(font_url, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page, probability_change_text_decoration_all_page, probability_change_font_style_all_page, probability_change_font_weight_all_page,
                        probability_change_font_variant_all_page, probability_change_font_stretch_all_page):
    if np.random.rand() < probability_change_text_color_all_page:
        text_color = "color: {}; ".format('#' + ''.join(np.random.choice(COLOR_HEX_RANGE, 6)))
    else:
        text_color = ""
    if np.random.rand() < probability_change_font_all_page:
        font_change = "font-family: data_generation_changing_font; "
    else:
        font_change = ""
    if np.random.rand() < probability_change_font_size_all_page:
        size_change = "font-size: {}px; ".format(np.random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1]))
    else:
        size_change = ""
    if np.random.rand() < probability_change_text_decoration_all_page:
        text_decoration = ""  # "text-decoration: {}; ".format(np.random.choice(TEXT_DECORATION_OPTIONS))
    else:
        text_decoration = ""
    if np.random.rand() < probability_change_font_style_all_page:
        style_change = "font-style: {}; ".format(np.random.choice(FONT_STYLE_OPTIONS))
    else:
        style_change = ""
    if np.random.rand() < probability_change_font_weight_all_page:
        weight_change = "font-weight: {}; ".format(np.random.choice(FONT_WEIGHT_OPTIONS))
    else:
        weight_change = ""
    if np.random.rand() < probability_change_font_variant_all_page:
        variant_change = "font-variant: {}; ".format(np.random.choice(FONT_VARIANT_OPTIONS))
    else:
        variant_change = ""
    if np.random.rand() < probability_change_font_stretch_all_page:
        stretch_change = "font-stretch: {}; ".format(np.random.choice(FONT_STRETCH_OPTIONS))
    else:
        stretch_change = ""
    js_code = '''
var newStyle = document.createElement('style');
newStyle.appendChild(document.createTextNode(\"@font-face {{ font-family: data_generation_changing_font; src: url(\'{}\'); }} body, div, p, pre, a, b, i, h1, h2, h3, h4, h5, h6, h7, span {{ {}{}{}{}{}{}{}{} }}\"));
document.head.appendChild(newStyle);
    '''.format(font_url, font_change, size_change, style_change, weight_change, variant_change, stretch_change, text_color, text_decoration)
    return js_code


def random_appearance_changing(driver, probability_change_background, font, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page,
                               probability_change_text_decoration_all_page, probability_change_font_style_all_page, probability_change_font_weight_all_page,
                               probability_change_font_variant_all_page, probability_change_font_stretch_all_page):
    background_changed = False
    if np.random.rand() < probability_change_background:
        driver.execute_script(CHANGE_BACKGROUND_JS_CODE)
        background_changed = True
    driver.execute_script(CHANGE_FONT_JS_CODE(font, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page,
                                              probability_change_text_decoration_all_page, probability_change_font_style_all_page, probability_change_font_weight_all_page,
                                              probability_change_font_variant_all_page, probability_change_font_stretch_all_page))
    return background_changed


def get_real_time_data(driver, screenshot_file, background_file, background_images_file, probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                       probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction='ltr', enable_icon_or_emoji_text=False, dictionary=None, probability_replace_word_from_dictionary=0.0,
                       screen_appearance_change_thr=0.05):
    tags_to_text = {}
    tags_to_bounding_boxes = {}
    tags_to_original_bounding_boxes = {}
    tags_to_images = {}
    tags_to_layout = {}
    WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(HIDE_OPEN_SHADOW_ROOTS))
    if sum([probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
            probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, probability_replace_word_from_dictionary]) > 0:
        [_, _, _, _, _, _] = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(EXTRACT_DATA_JS_CODE(probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                                                                                                                      probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction, enable_icon_or_emoji_text, dictionary, probability_replace_word_from_dictionary)))
        time.sleep(3.0)
    [word_ids, word_contents, word_bounding_boxes, image_bounding_boxes, words_layout, image_elements] = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(EXTRACT_DATA_JS_CODE(0, 0, 0, 0, 0,
                                                                                                                                                                                                  0, 0, 0, force_direction, enable_icon_or_emoji_text, dictionary, 0)))
    time.sleep(2.0)
    screen_data_before = base64.b64decode(driver.get_screenshot_as_base64())
    screen_img = Image.open(io.BytesIO(screen_data_before)).convert('RGB')
    screen_img.save(screenshot_file)  #, 'PNG', compress_level=0)
    word_idxs_didnt_pass = []
    for word_idx in tqdm(range(len(word_ids))):
        word_id, word_content, word_bounding_box = word_ids[word_idx], word_contents[word_idx], word_bounding_boxes[word_idx]
        element = driver.find_element(By.ID, word_id)
        try:
            element_image_before_change = Image.open(io.BytesIO(base64.b64decode(element.screenshot_as_base64))).convert('RGB')
            word_bounding_box = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(MAKE_ELEMENT_INVISIBLE_BY_ID, [word_id]))
            element_image_after_change = Image.open(io.BytesIO(base64.b64decode(element.screenshot_as_base64))).convert('RGB')
            mask_element_image = (np.abs(np.array(element_image_before_change.convert('RGB')) - np.array(element_image_after_change.convert('RGB'))).sum(axis=2) != 0).astype(np.uint8) * 255
        except Exception as e:
            raise e
        else:
            if np.any(mask_element_image > 0):
                y, x = np.where(mask_element_image > 0)
                resized_bounding_box = {
                    'top': int(np.round(y.min() + word_bounding_box['top'])),
                    'bottom': int(np.round(y.max() + word_bounding_box['top'])),
                    'left': int(np.round(x.min() + word_bounding_box['left'])),
                    'right': int(np.round(x.max() + word_bounding_box['left']))
                }
                tags_to_text[word_id] = word_content
                tags_to_bounding_boxes[word_id] = resized_bounding_box
                tags_to_original_bounding_boxes[word_id] = word_bounding_box
                tags_to_layout[word_id] = {
                    'reading_order': word_idx,
                    'text_node': words_layout[word_idx][0],
                    'tag_name': words_layout[word_idx][1],
                    'paragraph': words_layout[word_idx][2].id,
                    'paragraph_tag_name': words_layout[word_idx][3]
                }
            else:
                word_idxs_didnt_pass.append(word_idx)
    screen_data_background = io.BytesIO(base64.b64decode(driver.get_screenshot_as_base64()))
    screen_img_background = Image.open(screen_data_background).convert('RGB')
    screen_img_background.save(background_file)  #, 'PNG', compress_level=0)
    for word_idx in tqdm(range(len(word_ids))):
        if word_idx not in word_idxs_didnt_pass:
            word_id, word_content, word_bounding_box = word_ids[word_idx], word_contents[word_idx], word_bounding_boxes[word_idx]
            WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(MAKE_ELEMENT_VISIBLE_BY_ID, [word_id]))
    [word_ids_after, word_contents_after, word_bounding_boxes_after, image_bounding_boxes_after, words_layout_after, image_elements_after] = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(EXTRACT_DATA_JS_CODE(probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                                                                                                                                                                                                                                      probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction, enable_icon_or_emoji_text, dictionary, probability_replace_word_from_dictionary)))
    screen_img_after = np.array(Image.open(io.BytesIO(base64.b64decode(driver.get_screenshot_as_base64()))).convert('RGB'))
    y_changed, x_changed, c_changed = np.where(np.abs(np.array(screen_img) - screen_img_after) > 0)
    if (word_contents != word_contents_after) or (word_bounding_boxes != word_bounding_boxes_after) or (image_bounding_boxes != image_bounding_boxes_after):
        print('Num words : {}, {}'.format(len(word_ids), len(word_ids_after)))
        raise Exception('Animation!')
    if len(y_changed) >= screen_appearance_change_thr * screen_img_after.size:
        print('Screenshot changed!')
        raise Exception('Animation!')
    for image_idx in range(len(image_bounding_boxes)):
        image_bb, image_elem = image_bounding_boxes[image_idx], image_elements[image_idx]
        try:
            element_image_before_change = Image.open(io.BytesIO(base64.b64decode(image_elem.screenshot_as_base64))).convert('RGB')
            image_bb = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(MAKE_ELEMENT_INVISIBLE, image_elem))
            element_image_after_change = Image.open(io.BytesIO(base64.b64decode(image_elem.screenshot_as_base64))).convert('RGB')
            mask_element_image = (np.abs(np.array(element_image_before_change.convert('RGB')) - np.array(element_image_after_change.convert('RGB'))).sum(axis=2) != 0).astype(np.uint8) * 255
        except Exception as e:
            raise e
        else:
            if np.any(mask_element_image > 0):
                y, x = np.where(mask_element_image > 0)
                resized_bounding_box = {
                    'top': int(np.round(y.min() + max(int(np.round(image_bb['top'])), 0))),
                    'bottom': int(np.round(y.max() + max(int(np.round(image_bb['top'])), 0))),
                    'left': int(np.round(x.min() + max(int(np.round(image_bb['left'])), 0))),
                    'right': int(np.round(x.max() + max(int(np.round(image_bb['left'])), 0)))
                }
                tags_to_images['a' + str(uuid4())] = resized_bounding_box
    screen_data_images_background = io.BytesIO(base64.b64decode(driver.get_screenshot_as_base64()))
    screen_img_images_background = Image.open(screen_data_images_background).convert('RGB')
    screen_img_images_background.save(background_images_file)  #, 'PNG', compress_level=0)
    return tags_to_text, tags_to_bounding_boxes, tags_to_original_bounding_boxes, tags_to_images, tags_to_layout


@timeout_decorator.timeout(60)
def get_url(driver, url):
    if url[:4] != 'http':
        driver.get('file://' + os.path.abspath(url))
    else:
        driver.get(url)


def activate_html_and_get_data(html_file, screenshot_file, background_file, background_images_file, chromedriver, probability_random_scrolling, probability_change_background,
                               font, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page, probability_change_text_decoration_all_page, probability_change_font_style_all_page, probability_change_font_weight_all_page,
                               probability_change_font_variant_all_page, probability_change_font_stretch_all_page, char_level, probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                               probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction, window_size, log, enable_icon_or_emoji_text, dictionary,
                               probability_replace_word_from_dictionary, screen_appearance_change_thr, probability_change_by_web_modifier, webpage_modifier):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-web-security")
    options.add_argument("--start-maximized")
    options.add_argument("--headless")
    options.add_argument("--disable-notifications")
    options.add_argument("--hide-scrollbars")
    options.add_argument("window-size={},{}".format(window_size[0], window_size[1]))
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation", "disable-popup-blocking"])
    # options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 1,
    #                                           "profile.default_content_setting_values.geolocation": 1})
    driver = webdriver.Chrome(options=options)  #(executable_path=chromedriver, options=options)
    get_url(driver, html_file)
    try:
        driver.set_window_size(window_size[0], window_size[1])
    except Exception as e:
        print(str(e))
    window_size = driver.get_window_size()
    window_size = (window_size['height'], window_size['width'])
    if np.random.rand() < probability_random_scrolling:
        try:
            max_scroll = WebDriverWait(driver, timeout=2).until(lambda d: d.execute_script("return document.body.scrollHeight;"))
            random_height = np.random.randint(0, int(max_scroll) + 1)
            driver.execute_script("window.scrollTo(0, {});".format(random_height))
        except Exception as e:
            print(str(e))
    time.sleep(0.75)
    background_changed = random_appearance_changing(driver, probability_change_background, font, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page,
                                                    probability_change_text_decoration_all_page, probability_change_font_style_all_page, probability_change_font_weight_all_page,
                                                    probability_change_font_variant_all_page, probability_change_font_stretch_all_page)
    time.sleep(1.0)
    if np.random.rand() < probability_change_by_web_modifier:
        webpage_modifier.modify(driver)
        time.sleep(3.0)
    driver.execute_script("window.stop();")
    tags_to_text, tags_to_bounding_boxes, tags_to_original_bounding_boxes, tags_to_images, tags_to_layout = get_real_time_data(driver, screenshot_file, background_file, background_images_file, probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                                                                                                                               probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction, enable_icon_or_emoji_text, dictionary, probability_replace_word_from_dictionary,
                                                                                                                               screen_appearance_change_thr)
    driver.close()
    driver.quit()
    return tags_to_text, tags_to_bounding_boxes, tags_to_original_bounding_boxes, tags_to_images, tags_to_layout, window_size, background_changed


def merge_bounding_boxes_without_space_on_x_axis(tags_to_text, tags_to_bounding_boxes, tags_to_original_bounding_boxes, tags_to_layout, vertical_iou_merging_threshold):
    merged_all = False
    tags_that_changed = set()
    while not merged_all:
        merged_all = True
        rights_to_tags = {}
        lefts_to_tags = {}
        for tag, bounding_box in tags_to_original_bounding_boxes.items():
            top, left, bottom, right = bounding_box['top'], bounding_box['left'], bounding_box['bottom'], bounding_box['right']
            if right not in rights_to_tags.keys():
                rights_to_tags[right] = [tag]
            else:
                rights_to_tags[right].append(tag)
            if left not in lefts_to_tags.keys():
                lefts_to_tags[left] = [tag]
            else:
                lefts_to_tags[left].append(tag)
        iter_right_pos = list(rights_to_tags.keys()).copy()
        for right in iter_right_pos:
            iter_pos_tags = rights_to_tags[right].copy()
            for tag_right in iter_pos_tags:
                if right in lefts_to_tags.keys():
                    for tag_left in lefts_to_tags[right]:
                        if tag_right in tags_to_original_bounding_boxes.keys() and tag_left in tags_to_original_bounding_boxes.keys():
                            iou = vertical_1D_iou(tags_to_original_bounding_boxes[tag_right], tags_to_original_bounding_boxes[tag_left])
                            if iou >= vertical_iou_merging_threshold:
                                merged_all = False
                                tags_to_text[tag_left] = tags_to_text[tag_right] + tags_to_text[tag_left]
                                tags_to_bounding_boxes[tag_left] = {
                                    'top': min(tags_to_bounding_boxes[tag_right]['top'],
                                               tags_to_bounding_boxes[tag_left]['top']),
                                    'left': tags_to_bounding_boxes[tag_right]['left'],
                                    'bottom': max(tags_to_bounding_boxes[tag_right]['bottom'],
                                                  tags_to_bounding_boxes[tag_left]['bottom']),
                                    'right': tags_to_bounding_boxes[tag_left]['right']
                                }
                                del tags_to_text[tag_right]
                                del tags_to_bounding_boxes[tag_right]
                                del tags_to_original_bounding_boxes[tag_right]
                                del tags_to_layout[tag_right]
                                tags_that_changed.add(tag_left)
                                tags_that_changed.add(tag_right)
    return tags_to_text, tags_to_bounding_boxes, tags_to_layout, tags_that_changed


def filter_images_overlapping_text(tags_to_bounding_boxes, tags_to_images):
    images_tags = list(tags_to_images.keys()).copy()
    tags_to_images_not_overlapping_text = tags_to_images.copy()
    for image_tag in images_tags:
        for text_tag in tags_to_bounding_boxes.keys():
            if iou_2D(tags_to_images[image_tag], tags_to_bounding_boxes[text_tag]) > 0:
                del tags_to_images_not_overlapping_text[image_tag]
                break
    return tags_to_images_not_overlapping_text


def save_mask(screenshot_file, background_file, mask_file, background_images_file, mask_images_file):
    screenshot_img = Image.open(screenshot_file).convert("RGB")
    colorize_img = Image.open(background_file).convert("RGB")
    mask_image = Image.fromarray((np.abs(
        np.array(screenshot_img.convert('RGB')) - np.array(colorize_img.convert('RGB'))).sum(axis=2) != 0).astype(
        np.uint8) * 255)
    mask_image.save(mask_file, 'PNG')  #, compress_level=0)
    colorize_images_img = Image.open(background_images_file).convert("RGB")
    mask_images_image = Image.fromarray((np.abs(
        np.array(screenshot_img.convert('RGB')) - np.array(colorize_images_img.convert('RGB'))).sum(axis=2) != 0).astype(
        np.uint8) * 255)
    mask_images_image.save(mask_images_file, 'PNG')  #, compress_level=0)


def save_bounding_box_mask(tags_to_bounding_boxes, tags_to_images, mask_file, bounding_boxes_text_mask_file, bounding_boxes_text_mask_transformed_file, bounding_boxes_image_mask_file, bounding_boxes_image_mask_transformed_file):
    mask_img = np.array(Image.open(mask_file).convert("RGB")).sum(axis=2)
    bounding_boxes_mask = np.zeros(mask_img.shape)
    bounding_boxes_mask_invert = np.ones(mask_img.shape)
    for bounding_box in tags_to_bounding_boxes.values():
        bounding_boxes_mask[bounding_box['top']: bounding_box['bottom'] + 1,
        bounding_box['left']: bounding_box['right'] + 1] = 255
        bounding_boxes_mask_invert[bounding_box['top'] - 1: bounding_box['bottom'] + 2,
        bounding_box['left'] - 1: bounding_box['right'] + 2] = 0
    bounding_boxes_image_mask = np.zeros(mask_img.shape)
    for bounding_box in tags_to_images.values():
        bounding_boxes_image_mask[bounding_box['top']: bounding_box['bottom'] + 1,
        bounding_box['left']: bounding_box['right'] + 1] = 255
    bounding_boxes_image_mask = Image.fromarray(bounding_boxes_image_mask).convert('RGB')
    bounding_boxes_image_mask.save(bounding_boxes_image_mask_file)  #, compress_level=0)
    bounding_boxes_image_mask.save(bounding_boxes_image_mask_transformed_file)  #, compress_level=0)
    ones_y, ones_x = np.where(bounding_boxes_mask == 255)
    num_ones = len(ones_y)
    num_zeros = bounding_boxes_mask.shape[0] * bounding_boxes_mask.shape[1] - num_ones
    bounding_boxes_mask = Image.fromarray(bounding_boxes_mask).convert('RGB')
    bounding_boxes_mask.save(bounding_boxes_text_mask_file)  #, compress_level=0)
    bounding_boxes_mask.save(bounding_boxes_text_mask_transformed_file)  #, compress_level=0)
    return [num_zeros, num_ones]


def generate_data_to_recognition(tags_to_text, tags_to_bounding_boxes, tags_that_changed, screenshot_file, mask_file, recognition_folder, json_file, force_direction):
    screenshot_img = np.array(Image.open(screenshot_file).convert("RGB"))
    mask_img = np.array(Image.open(mask_file).convert("RGB"))
    num_samples = 0
    file_names_to_data = {}
    bounding_boxes_and_text_for_recognition = []
    for tag, word in tags_to_text.items():
        if tag in tags_to_bounding_boxes.keys() and tag not in tags_that_changed:
            bounding_box = tags_to_bounding_boxes[tag]
            bounding_boxes_and_text_for_recognition.append((word, bounding_box))
            word_img = Image.fromarray(screenshot_img[bounding_box['top']: bounding_box['bottom'] + 1,
                                       bounding_box['left']: bounding_box['right'] + 1, :]).convert('RGB')
            word_mask_img = Image.fromarray(mask_img[bounding_box['top']: bounding_box['bottom'] + 1,
                                            bounding_box['left']: bounding_box['right'] + 1, :]).convert('RGB')
            file_name = str(num_samples) + '.jpg'
            mask_file_path = os.path.join(recognition_folder, str(num_samples) + '_mask.png')
            word_img.save(os.path.join(recognition_folder, file_name))  #, 'PNG', compress_level=0)
            word_mask_img.save(mask_file_path, 'PNG')  #, compress_level=0)
            file_names_to_data[file_name] = {
                'text': word,
                'image_size': (word_img.size[1], word_img.size[0]),
                'rotated': False,
                'mask_file': mask_file_path,
                'direction': str(force_direction)
            }
            num_samples += 1
    save_to_json(file_names_to_data, json_file)
    return bounding_boxes_and_text_for_recognition


def save_json(tags_to_text, tags_to_bounding_boxes, bounding_boxes_and_text_for_recognition,
              tags_to_images, tags_to_images_not_overlapping_text, window_size,
              mask_file, mask_images_file, bounding_boxes_text_mask_file, bounding_boxes_text_mask_transformed_file,
              bounding_boxes_image_mask_file, bounding_boxes_image_mask_transformed_file, background_file, background_images_file,
              num_pixels_each_class, json_file, html_file, background_changed, layout, json_layout_file, generate_layout_data, force_direction):
    data = {
        'tags_to_text': tags_to_text,
        'tags_to_bounding_boxes_text': tags_to_bounding_boxes,
        'bounding_boxes_and_text_for_recognition': bounding_boxes_and_text_for_recognition,
        'tags_to_bounding_boxes_images': tags_to_images,
        'tags_to_bounding_boxes_images_not_overlapping_text': tags_to_images_not_overlapping_text,
        'tags_to_bounding_boxes_text_original': tags_to_bounding_boxes,
        'bounding_boxes_and_text_for_recognition_original': bounding_boxes_and_text_for_recognition,
        'tags_to_bounding_boxes_images_original': tags_to_images,
        'tags_to_bounding_boxes_images_not_overlapping_text_original': tags_to_images_not_overlapping_text,
        'image_size': window_size,
        'mask_file': mask_file,
        'mask_images_file': mask_images_file,
        'data_file': json_file,
        'bounding_boxes_text_mask_file': bounding_boxes_text_mask_file,
        'bounding_boxes_text_mask_transformed_file': bounding_boxes_text_mask_transformed_file,
        'bounding_boxes_image_mask_file': bounding_boxes_image_mask_file,
        'bounding_boxes_image_mask_transformed_file': bounding_boxes_image_mask_transformed_file,
        'background_file': background_file,
        'background_images_file': background_images_file,
        'num_pixels_each_class': num_pixels_each_class,
        'perspective': None,
        'elastic': None,
        'url': html_file,
        'background_changed': background_changed,
        'direction': str(force_direction)
    }
    save_to_json(data, json_file)
    if generate_layout_data:
        save_to_json(layout, json_layout_file)


def divide_first_overlapping_paragraph(paragraph1, paragraph2):
    for ibbi, bbi in enumerate(paragraph1[1][:-1]):
        group1_a = paragraph1[1][:ibbi + 1]
        group1_b = paragraph1[1][ibbi + 1:]
        group1_a_bb = {
            'left': min([x[1]['left'] for x in group1_a]),
            'top': min([x[1]['top'] for x in group1_a]),
            'right': max([x[1]['right'] for x in group1_a]),
            'bottom': max([x[1]['bottom'] for x in group1_a])
        }
        group1_b_bb = {
            'left': min([x[1]['left'] for x in group1_b]),
            'top': min([x[1]['top'] for x in group1_b]),
            'right': max([x[1]['right'] for x in group1_b]),
            'bottom': max([x[1]['bottom'] for x in group1_b])
        }
        if iou_2D(group1_a_bb, paragraph2[0]) == 0.0 and iou_2D(group1_b_bb, paragraph2[0]) == 0.0:
            paragraph1a = (group1_a_bb, group1_a)
            paragraph1b = (group1_b_bb, group1_b)
            return (True, (paragraph1a, paragraph1b, paragraph2))
    return (False, None)


def handle_overlapping_paragraphs(paragraphs):
    deleted_paragraphs = []
    added_paragraphs = []
    loops_over = ['contain', 'overlap']
    for loop in loops_over:
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                if paragraphs[i] not in deleted_paragraphs and paragraphs[j] not in deleted_paragraphs:
                    iou = iou_2D(paragraphs[i][0], paragraphs[j][0])
                    if loop == 'contain':
                        area_i = area_2D(paragraphs[i][0])
                        area_j = area_2D(paragraphs[j][0])
                        if area_i >= area_j and iou >= 0.95 * area_j:
                            deleted_paragraphs.append(paragraphs[j])
                        elif area_j >= area_i and iou >= 0.95 * area_i:
                            deleted_paragraphs.append(paragraphs[i])
                    else:
                        if iou > 0:
                            div_i = divide_first_overlapping_paragraph(paragraphs[i], paragraphs[j])
                            if div_i[0]:
                                added_paragraphs.append(div_i[1][0])
                                added_paragraphs.append(div_i[1][1])
                                deleted_paragraphs.append(paragraphs[i])
                            div_j = divide_first_overlapping_paragraph(paragraphs[j], paragraphs[i])
                            if div_j[0]:
                                added_paragraphs.append(div_j[1][0])
                                added_paragraphs.append(div_j[1][1])
                                deleted_paragraphs.append(paragraphs[j])
    return deleted_paragraphs, added_paragraphs


def generate_layout(tags_to_text, tags_to_bounding_boxes, tags_to_layout):
    sorted_tags = [t for t, _ in sorted(tags_to_layout.items(), key=lambda x: x[1]['reading_order'])]
    sorted_tags_relevant_paragraphs = []
    for i in range(len(sorted_tags)):
        i_until = i
        for j in range(i + 1, len(sorted_tags)):
            if tags_to_layout[sorted_tags[j]]['paragraph'] == tags_to_layout[sorted_tags[i]]['paragraph']:
                i_until = j
        sorted_tags_relevant_paragraphs.append(i_until)
    paragraphs = []
    last_paragraph = []
    last_paragraph_ending = -1
    for itag, tag in enumerate(sorted_tags):
        curr_bb = tags_to_bounding_boxes[tag]
        curr_text = tags_to_text[tag]
        if last_paragraph_ending >= itag:
            start_new_paragraph = False
        else:
            last_paragraph_ending = sorted_tags_relevant_paragraphs[itag]
            start_new_paragraph = True
        if start_new_paragraph and len(last_paragraph) > 0:
            paragraphs.append(last_paragraph)
            last_paragraph = []
        last_paragraph.append((tag, curr_bb, curr_text))
    paragraphs.append(last_paragraph)
    for i in range(len(paragraphs)):
        paragraphs[i] = ({
                             'left': min([x[1]['left'] for x in paragraphs[i]]),
                             'top': min([x[1]['top'] for x in paragraphs[i]]),
                             'right': max([x[1]['right'] for x in paragraphs[i]]),
                             'bottom': max([x[1]['bottom'] for x in paragraphs[i]])
                         }, paragraphs[i])
    deleted_paragraphs = [-1]
    added_paragraphs = [-1]
    loops = 3
    while loops > 0 and (len(added_paragraphs) != 0 or len(deleted_paragraphs) != 0):
        deleted_paragraphs, added_paragraphs = handle_overlapping_paragraphs(paragraphs)
        for p in deleted_paragraphs:
            if p in paragraphs:
                paragraphs.remove(p)
        paragraphs += added_paragraphs
        loops -= 1
    layout = {
        'paragraphs': paragraphs
    }
    return layout


def html2OcrData(html_src, dst_folder, vertical_iou_merging_threshold,
                 probability_random_scrolling, probability_change_background, font,
                 probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page, probability_change_text_decoration_all_page,
                 probability_change_font_style_all_page, probability_change_font_weight_all_page,
                 probability_change_font_variant_all_page, probability_change_font_stretch_all_page, probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
                 probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction,
                 char_level, chromedriver, window_size, log, ready_data, enable_icon_or_emoji_text, dictionary, probability_replace_word_from_dictionary,
                 generate_layout_data, screen_appearance_change_thr, probability_change_by_web_modifier, webpage_modifier):
    shutil.rmtree(dst_folder, ignore_errors=True)
    os.makedirs(dst_folder, exist_ok=True)
    detection_folder = os.path.join(dst_folder, 'detection_data')
    recognition_folder = os.path.join(dst_folder, 'recognition_data')
    os.makedirs(detection_folder, exist_ok=True)
    os.makedirs(recognition_folder, exist_ok=True)
    layout_folder = os.path.join(dst_folder, 'layout_data')
    if generate_layout_data:
        os.makedirs(layout_folder, exist_ok=True)
    json_layout_file = os.path.join(layout_folder, 'data.json')
    screenshot_file = os.path.join(detection_folder, 'image.jpg')
    mask_file = os.path.join(detection_folder, 'mask.png')
    mask_images_file = os.path.join(detection_folder, 'mask_images.png')
    bounding_boxes_text_mask_file = os.path.join(detection_folder, 'bounding_boxes_words_mask.png')
    bounding_boxes_text_mask_transformed_file = os.path.join(detection_folder, 'bounding_boxes_words_mask_transformed.png')
    bounding_boxes_image_mask_file = os.path.join(detection_folder, 'bounding_boxes_images_mask.png')
    bounding_boxes_image_mask_transformed_file = os.path.join(detection_folder, 'bounding_boxes_images_mask_transformed.png')
    background_file = os.path.join(detection_folder, 'background.jpg')
    background_images_file = os.path.join(detection_folder, 'background_images.jpg')
    json_detection_file = os.path.join(detection_folder, 'data.json')
    json_recognition_file = os.path.join(recognition_folder, 'data.json')
    tags_to_text, tags_to_bounding_boxes, tags_to_original_bounding_boxes, tags_to_images, tags_to_layout, window_size, background_changed = activate_html_and_get_data(
        html_src, screenshot_file, background_file, background_images_file, chromedriver, probability_random_scrolling,
        probability_change_background, font, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page,
        probability_change_text_decoration_all_page, probability_change_font_style_all_page, probability_change_font_weight_all_page,
        probability_change_font_variant_all_page, probability_change_font_stretch_all_page, char_level, probability_change_text_color, probability_change_font, probability_change_font_size, probability_change_text_decoration, probability_change_font_style,
        probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch, force_direction, window_size, log, enable_icon_or_emoji_text,
        dictionary, probability_replace_word_from_dictionary, screen_appearance_change_thr, probability_change_by_web_modifier, webpage_modifier)
    tags_to_text, tags_to_bounding_boxes, tags_to_layout, tags_that_changed = merge_bounding_boxes_without_space_on_x_axis(tags_to_text, tags_to_bounding_boxes, tags_to_original_bounding_boxes, tags_to_layout, vertical_iou_merging_threshold)
    tags_to_images_not_overlapping_text = filter_images_overlapping_text(tags_to_bounding_boxes, tags_to_images)
    save_mask(screenshot_file, background_file, mask_file, background_images_file, mask_images_file)
    num_pixels_each_class = save_bounding_box_mask(tags_to_bounding_boxes, tags_to_images, mask_file, bounding_boxes_text_mask_file, bounding_boxes_text_mask_transformed_file, bounding_boxes_image_mask_file, bounding_boxes_image_mask_transformed_file)
    bounding_boxes_and_text_for_recognition = generate_data_to_recognition(tags_to_text, tags_to_bounding_boxes, tags_that_changed, screenshot_file, mask_file, recognition_folder, json_recognition_file, force_direction)
    if generate_layout_data:
        layout = generate_layout(tags_to_text, tags_to_bounding_boxes, tags_to_layout)
        layout['image_size'] = window_size
        shutil.copyfile(screenshot_file, os.path.join(layout_folder, 'image.jpg'))
    else:
        layout = None
    save_json(tags_to_text, tags_to_bounding_boxes, bounding_boxes_and_text_for_recognition, tags_to_images,
              tags_to_images_not_overlapping_text, window_size,
              mask_file, mask_images_file, bounding_boxes_text_mask_file, bounding_boxes_text_mask_transformed_file, bounding_boxes_image_mask_file, bounding_boxes_image_mask_transformed_file, background_file, background_images_file,
              num_pixels_each_class, json_detection_file, html_src, background_changed, layout, json_layout_file, generate_layout_data, force_direction)
    # ready_data.make_detection_data_ready([detection_folder])
    convert_detection_to_coco(detection_folder)
    if generate_layout_data:
        convert_layout_to_coco(layout_folder)
    return screenshot_file, tags_to_text, tags_to_bounding_boxes, tags_to_images, tags_to_images_not_overlapping_text, layout


def draw_demonstration(screenshot_file_before, screenshot_file_after, demonstration_layout_file, tags_to_text, tags_to_bounding_boxes,
                       tags_to_images, layout, font_file, generate_layout_data):
    screenshot = Image.open(screenshot_file_before).convert('RGB')
    draw = ImageDraw.Draw(screenshot)
    font = ImageFont.truetype(font_file, 12, encoding='utf-8')
    for tag in tags_to_bounding_boxes.keys():
        top, left, bottom, right = tags_to_bounding_boxes[tag]['top'], tags_to_bounding_boxes[tag]['left'], \
            tags_to_bounding_boxes[tag]['bottom'], tags_to_bounding_boxes[tag]['right']
        if tag in tags_to_text.keys():
            draw.rectangle([(left, top), (right, bottom)], outline='blue')
            draw.text((left, top - 10), tags_to_text[tag], font=font, fill='blue')
        else:
            draw.rectangle([(left, top), (right, bottom)], outline='green')
    for tag in tags_to_images.keys():
        top, left, bottom, right = tags_to_images[tag]['top'], \
            tags_to_images[tag]['left'], \
            tags_to_images[tag]['bottom'], \
            tags_to_images[tag]['right']
        draw.rectangle([(left, top), (right, bottom)], outline='red')
        draw.text((left, top - 10), 'IMG', font=font, fill='red')
    screenshot.save(screenshot_file_after)  #, 'PNG', compress_level=0)
    if generate_layout_data:
        layout_screen = Image.open(screenshot_file_before).convert('RGB')
        layout_draw = ImageDraw.Draw(layout_screen)
        for p in layout['paragraphs']:
            layout_draw.rectangle(((p[0]['left'], p[0]['top']), (p[0]['right'], p[0]['bottom'])), outline='red')
        layout_screen.save(demonstration_layout_file)  #, compress_level=0)


class WebTemplateModifier:

    def __init__(self, fonts_dir, corpuses_dir, modifications_to_probs=None):
        self.modifications_to_probs = modifications_to_probs
        self.fonts = [os.path.join(os.getcwd(), fonts_dir, fn) for fn in os.listdir(fonts_dir) if '.ttf' in fn.lower()]
        self.corpuses_dir = corpuses_dir
        self.corpuses, self.corpuses_probs, self.corpuses_paths_to_corpuses_names = self._get_corpuses()
        self.corpuses_idxs = list(range(len(self.corpuses)))

    @timeout_decorator.timeout(80)
    def modify(self, driver):
        text_node_lengths = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(self._text_node_modification_code()))
        if self.modifications_to_probs is None or ('text' in self.modifications_to_probs.keys() and np.random.rand() < self.modifications_to_probs['text']):
            self._modify_text(driver, text_node_lengths)
        if self.modifications_to_probs is None or ('font_all_page' in self.modifications_to_probs.keys() and np.random.rand() < self.modifications_to_probs['font_all_page']):
            self._change_font_all_page(driver)
        elif 'font_each_element' in self.modifications_to_probs.keys() and np.random.rand() < self.modifications_to_probs['font_each_element']:
            self._change_font_each_element(driver, text_node_lengths)

    def _modify_text(self, driver, text_node_lengths):
        corpus_idx = np.random.choice(self.corpuses_idxs, p=self.corpuses_probs)
        json_data = json.load(open(self.corpuses[corpus_idx], mode='r', encoding='utf-8'))
        dst_strings = [self._get_random_text(json_data, string_length) for string_length in text_node_lengths]
        text_node_lengths2 = WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(self._text_node_modification_code(dst_strings)))
        assert text_node_lengths == text_node_lengths2

    def _modify_font(self, driver, text_node_lengths):
        self._change_font_all_page(driver)
        self._change_font_each_element(driver, text_node_lengths)

    def _get_corpuses(self):
        corpuses = []
        json_paths, json_probs = [], []
        corpuses_paths_to_corpuses_names = {}
        total_num_paragraphs = 0.0
        data_json_path = os.path.join(self.corpuses_dir, 'data.json')
        if os.path.isfile(data_json_path):
            json_data = json.load(open(data_json_path, mode='r', encoding='utf-8'))
            corpuses_paths, corpuses_probs = json_data['corpuses_paths'], json_data['corpuses_probs']
            json_paths += corpuses_paths
            json_probs += corpuses_probs
        else:
            for corpus_json_file in os.listdir(self.corpuses_dir):
                if corpus_json_file.split('.')[-1].lower() == 'json':
                    corpus_json_path = os.path.join(self.corpuses_dir, corpus_json_file)
                    json_data = json.load(open(corpus_json_path, mode='r', encoding='utf-8'))
                    num_paragraphs = len(json_data)
                    total_num_paragraphs += num_paragraphs
                    corpuses.append([corpus_json_path, num_paragraphs])
            corpuses_paths, corpuses_probs = [i[0] for i in corpuses], [i[1] / total_num_paragraphs for i in corpuses]
            json_paths += corpuses_paths
            json_probs += corpuses_probs
            save_to_json({'corpuses_paths': corpuses_paths, 'corpuses_probs': corpuses_probs}, data_json_path)
        corpuses_paths_to_corpuses_names.update({k: 'corpuses' for k in corpuses_paths})
        json_probs_sum = sum(json_probs)
        json_probs = [i / json_probs_sum for i in json_probs]
        return json_paths, json_probs, corpuses_paths_to_corpuses_names

    def _get_random_text(self, json_data, string_length):
        par = []
        while True:
            idx = np.random.randint(0, len(json_data))
            sen = json_data[idx]['text'].split(' ')
            if np.random.rand() < 0.5 and len(sen) > 1:
                sen = sen[1:]
            par = par + sen
            for iw in range(1, len(par)):
                pat_str = self._list_to_string(par[:iw])
                if len(pat_str) >= string_length:
                    return pat_str

    def _list_to_string(self, par_list):
        par = ' '.join(par_list)
        par = par.replace('\n', ' ')
        par = par.replace('\t', ' ')
        par = par.replace('{| |} ', '')
        return par

    def _text_node_modification_code(self, dst_strings=None, dst_ids=None):
        js_text_nodes = '''
    var lengths = [];
    var nodes = [];
    var warpping_nodes = [];
    const spaces = ['\\n', ' ', '\\t'];
    const dst_strings = {};
    const dst_ids = {};
    function find_text_nodes(node) {{
        for (var i=0; i < node.childNodes.length; i++) {{
            if (node.childNodes[i].nodeType === Node.TEXT_NODE && node.childNodes[i].textContent && node.childNodes[i].textContent.length > 0) {{
                var actual_string = false;
                for (var ch=0; ch < node.childNodes[i].textContent.length; ch++) {{
                    if (!spaces.includes(node.childNodes[i].textContent[ch])) {{
                        actual_string = true;
                    }}
                }}
                if (actual_string) {{
                    lengths.push(node.childNodes[i].textContent.length);
                    nodes.push(node.childNodes[i]);
                    warpping_nodes.push(node);
                }}
            }} else if (node.childNodes[i].nodeType === Node.ELEMENT_NODE) {{
                find_text_nodes(node.childNodes[i]);
            }}
        }}
    }}
    find_text_nodes(document);
    if (dst_strings != null) {{
        for (var i=0; i < nodes.length; i++) {{
            nodes[i].textContent = dst_strings[i];
        }}
    }}
    if (dst_ids != null) {{
        for (var i=0; i < nodes.length; i++) {{
            warpping_nodes[i].classList.add(dst_ids[i]);
        }}
    }}
    return lengths;
            '''.format('null' if dst_strings is None else dst_strings, 'null' if dst_ids is None else dst_ids)
        return js_text_nodes

    def _get_style_code(self, objects_to_style='body, div, p, pre, a, b, i, h1, h2, h3, h4, h5, h6, h7, span', return_true=True):
        font_url = np.random.choice(self.fonts)
        font_name = ''.join(font_url.split(os.sep)[-1].replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('_', ' ').split(' '))
        if np.random.rand() < 0.5:
            text_color = "color: {}; ".format('#' + ''.join(np.random.choice(COLOR_HEX_RANGE, 6)))
        else:
            text_color = ""
        if np.random.rand() < 0.5:
            font_change = "font-family: {}; ".format(font_name)
        else:
            font_change = ""
        if np.random.rand() < 0.5:
            size_change = "font-size: {}px; ".format(np.random.randint(10, 40))
        else:
            size_change = ""
        if np.random.rand() < 0.5:
            text_decoration = ""  # "text-decoration: {}; ".format(np.random.choice(TEXT_DECORATION_OPTIONS))
        else:
            text_decoration = ""
        if np.random.rand() < 0.5:
            style_change = "font-style: {}; ".format(np.random.choice(FONT_STYLE_OPTIONS))
        else:
            style_change = ""
        if np.random.rand() < 0.5:
            weight_change = "font-weight: {}; ".format(np.random.choice(FONT_WEIGHT_OPTIONS))
        else:
            weight_change = ""
        if np.random.rand() < 0.5:
            variant_change = "font-variant: {}; ".format(np.random.choice(FONT_VARIANT_OPTIONS))
        else:
            variant_change = ""
        if np.random.rand() < 0.5:
            letter_spacing_change = "letter-spacing: {}px; ".format(np.random.randint(LETTER_SPACING_RANGE[0], LETTER_SPACING_RANGE[1] + 1))
        else:
            letter_spacing_change = ""
        if np.random.rand() < 0.5:
            stretch_change = "font-stretch: {}; ".format(np.random.choice(FONT_STRETCH_OPTIONS))
        else:
            stretch_change = ""
        js_code_font = '''
var newFont = document.createElement('style');
newFont.appendChild(document.createTextNode(\"@font-face {{ font-family: {}; src: url(\'{}\'); }}\"));
document.head.appendChild(newFont);
'''.format(font_name, font_url)
                   # objects_to_style,
        if return_true:
            js_code_font += 'return true;\n'
        node_inline_style = "{}{}{}{}{}{}{}{}{}".format(font_change, size_change, style_change, weight_change, variant_change, letter_spacing_change, stretch_change, text_color, text_decoration)
        not_inline_style = '''
var newStyle = document.createElement('style');
newStyle.appendChild(document.createTextNode(\"@font-face {{ font-family: {}; src: url(\'{}\'); }} {} {{ {}{}{}{}{}{}{}{}{} }}\"));
document.head.appendChild(newStyle);
'''.format(font_name, font_url,
                   objects_to_style,
                   font_change, size_change, style_change, weight_change, variant_change, letter_spacing_change, stretch_change, text_color, text_decoration)
        if return_true:
            not_inline_style += 'return true;\n'
        return js_code_font, node_inline_style, not_inline_style

    def _change_font_all_page(self, driver):
        _, _, js_code = self._get_style_code()
        WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(js_code))

    def _change_font_each_element(self, driver, text_node_lengths):
        dst_ids = ['a' + str(uuid4()).replace('-', '') for _ in range(len(text_node_lengths))]
        WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(self._text_node_modification_code(dst_ids=dst_ids)))
        js_code_fonts = ''
        nodes_inline_styles = {}
        for dst_id in dst_ids:
            font_code, node_inline_style, _ = self._get_style_code('.' + dst_id, return_true=False)
            js_code_fonts += font_code
            nodes_inline_styles[dst_id] = node_inline_style
        js_code_fonts += 'return true;\n'
        WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script(js_code_fonts))
        for dst_id in dst_ids:
            node_inline_style = nodes_inline_styles[dst_id]
            WebDriverWait(driver, timeout=100).until(lambda d: d.execute_script("document.getElementsByClassName('{}')[0].style.cssText = \"{}\";\nreturn true;".format(dst_id, node_inline_style)))


def generate_ocr_data(html_paths,
                      dst_folder,
                      running_name,
                      font,
                      log,
                      num_samples=np.inf,
                      window_size=(1920, 1080),
                      vertical_iou_merging_threshold=0.5,
                      probability_random_scrolling=1.0,
                      probability_change_background=1.0,
                      probability_change_text_color_all_page=1.0,
                      probability_change_font_all_page=1.0,
                      probability_change_font_size_all_page=1.0,
                      probability_change_text_decoration_all_page=1.0,
                      probability_change_font_style_all_page=1.0,
                      probability_change_font_weight_all_page=1.0,
                      probability_change_font_variant_all_page=1.0,
                      probability_change_font_stretch_all_page=1.0,
                      probability_change_text_color=1.0,
                      probability_change_font=1.0,
                      probability_change_font_size=1.0,
                      probability_change_text_decoration=1.0,
                      probability_change_font_style=1.0,
                      probability_change_font_variant=1.0,
                      probability_change_font_weight=1.0,
                      probability_change_font_stretch=1.0,
                      force_direction='ltr',
                      char_level=False,
                      chromedriver=r'../../necessary_files/chromedriver',
                      rectangular_bounding_boxes=True,
                      detection_data_classes_field_names=('bounding_boxes_text',),
                      detection_data_fields_to_mask_in_loss=('bounding_boxes_images',),
                      consider_elastic=True,
                      consider_perspective=True,
                      enable_icon_or_emoji_text=False,
                      dictionary=None,
                      probability_replace_word_from_dictionary=0.0,
                      generate_layout_data=True,
                      screen_appearance_change_thr=0.05,
                      probability_change_by_web_modifier=0.0,
                      webpage_modifier=None
                      ):
    os.makedirs(dst_folder, exist_ok=True)
    running_folder = os.path.join(dst_folder, running_name)
    os.makedirs(running_folder, exist_ok=True)
    if num_samples == np.Inf:
        num_samples = len(html_paths)
    count = 0
    count_num_samples = 0
    used_urls = Counter()
    # ready_data = ReadyDetectionData(rectangular_bounding_boxes=rectangular_bounding_boxes, detection_data_classes_field_names=detection_data_classes_field_names, detection_data_fields_to_mask_in_loss=detection_data_fields_to_mask_in_loss, consider_elastic=consider_elastic, consider_affine=consider_affine)
    ready_data = None
    while count_num_samples < num_samples:
        start_time = time.time()
        html_path_idx = count % len(html_paths)
        html_path = html_paths[html_path_idx]
        dst_folder_datapoint = os.path.join(running_folder, str(uuid4()))
        demonstration_file = os.path.join(dst_folder_datapoint, 'demonstration.jpg')
        demonstration_layout_file = os.path.join(dst_folder_datapoint, 'demonstration_layout.jpg')
        try:
            if html_path[:4] == 'http':
                res = requests.get(html_path, timeout=3)
                if res.status_code != 200:
                    raise Exception('URL {} is invalid'.format(html_path))
            else:
                if not os.path.isfile(html_path):
                    raise Exception('The file {} is not existed'.format(html_path))
            try:
                print(html_path)
            except Exception as e:
                print(str(e))
            screenshot_file, tags_to_text, tags_to_bounding_boxes, tags_to_images, tags_to_images_not_overlapping_text, layout = html2OcrData(
                html_path, dst_folder_datapoint, vertical_iou_merging_threshold,
                probability_random_scrolling, probability_change_background, font, probability_change_text_color_all_page, probability_change_font_all_page, probability_change_font_size_all_page, probability_change_text_decoration_all_page,
                probability_change_font_style_all_page, probability_change_font_weight_all_page,
                probability_change_font_variant_all_page, probability_change_font_stretch_all_page,
                probability_change_text_color, probability_change_font, probability_change_font_size,
                probability_change_text_decoration, probability_change_font_style,
                probability_change_font_variant, probability_change_font_weight, probability_change_font_stretch,
                force_direction, char_level, chromedriver, window_size, log, ready_data, enable_icon_or_emoji_text, dictionary,
                probability_replace_word_from_dictionary, generate_layout_data, screen_appearance_change_thr, probability_change_by_web_modifier, webpage_modifier)
            draw_demonstration(screenshot_file, demonstration_file, demonstration_layout_file, tags_to_text, tags_to_bounding_boxes,
                               tags_to_images, layout, font, generate_layout_data)
            print(time.time() - start_time)
            used_urls.update([html_path])
            count_num_samples += 1
            count += 1
        except Exception as e:
            print(str(e))
            shutil.rmtree(dst_folder_datapoint, ignore_errors=True)
            count += 1
            continue
        running_info_file = os.path.join(running_folder, 'running_info.json')
        running_info = {
            'html_paths': html_paths,
            'dst_folder': dst_folder,
            'running_name': running_name,
            'font_view_name': font,
            'num_samples': count,
            'window_size': window_size,
            'vertical_iou_merging_threshold': vertical_iou_merging_threshold,
            'probability_random_scrolling': probability_random_scrolling,
            'probability_change_font': probability_change_font,
            'probability_change_background': probability_change_background,
            'force_direction': force_direction,
            'char_level': char_level,
            'chromedriver': chromedriver,
            'rectangular_bounding_boxes': rectangular_bounding_boxes,
            'detection_data_classes_field_names': detection_data_classes_field_names,
            'detection_data_fields_to_mask_in_loss': detection_data_fields_to_mask_in_loss,
            'consider_elastic': consider_elastic,
            'consider_perspective': consider_perspective,
            'used_urls': dict(used_urls),
            'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
            'dictionary_view_name': dictionary,
            'probability_replace_word_from_dictionary': probability_replace_word_from_dictionary,
            'dictionary': str(dictionary),
            'generate_layout_data': generate_layout_data,
            'screen_appearance_change_thr': screen_appearance_change_thr,
            'probability_change_by_web_modifier': probability_change_by_web_modifier,
            # 'webpage_modifier_corpuses': webpage_modifier.text_corpus_names
        }
        save_to_json(running_info, running_info_file)
    convert_detection_to_coco(running_folder)
    if generate_layout_data:
        convert_layout_to_coco(running_folder)


if __name__ == '__main__':
    #  To compile: pyinstaller --onefile -F v1.py
    all_page_prob = 0.0
    words_prob = 0.0
    if sys.argv[1] in ['-h', '--help']:
        print('./v1 running_name dst_folder chromedriver web_links_json_file probability_change_background probability_change_text probability_change_all_page_font probability_change_each_element_font direction complete_only_missing_urls')
    else:
        print('This program using JPEG compression built-in for compressing the generated data.')
        running_name = sys.argv[1]
        dst_folder = sys.argv[2]
        chromedriver = sys.argv[3]
        web_links_json_file = sys.argv[4]
        font = [os.path.join('fonts', fn) for fn in os.listdir('fonts') if '.ttf' in fn.lower()][0]
        probability_change_background = float(sys.argv[5])
        probability_change_text = float(sys.argv[6])
        probability_change_all_page_font = float(sys.argv[7])
        probability_change_each_element_font = float(sys.argv[8])
        force_direction = str(sys.argv[9])
        complete_only_missing_urls = sys.argv[10].lower() == 'true'
        modifier = WebTemplateModifier('fonts', 'corpuses', {'text': probability_change_text, 'font_all_page': probability_change_all_page_font,
                                                             'font_each_element': probability_change_each_element_font})
        web_links = json.load(open(web_links_json_file, mode='r', encoding='utf-8'))
        web_links = [i for v in web_links.values() for i in v]
        if complete_only_missing_urls:  # For now, supported only for the standalone version
            web_links = [i for idx, i in enumerate(web_links) if i not in web_links[:idx]]
            idx_last = -1
            running_dir = os.path.join(dst_folder, running_name)
            if os.path.isdir(running_dir):
                already_used_urls = []
                for fold in os.listdir(running_dir):
                    data_json = os.path.join(running_dir, fold, 'detection_data', 'data.json')
                    if os.path.isfile(data_json):
                        url = json.load(open(data_json, mode='r', encoding='utf-8'))['url']
                        if url in web_links:
                            idx_last = max(idx_last, web_links.index(url))
                            already_used_urls.append(url)
                            web_links.remove(url)
                            print('Already used the following url : {}'.format(url))
                print('Starting from url number {}'.format(idx_last + 1))
                # web_links = web_links[idx_last + 1:] if idx_last < (len(web_links) - 1) else []
                web_links.sort(key=lambda x: np.random.rand())
        kwargs = {
            'html_paths': web_links,
            'dst_folder': dst_folder,
            'running_name': running_name,
            'chromedriver': chromedriver,
            'font': font,
            'log': None,
            'num_samples': np.inf,
            'window_size': (1920, 1080),
            'vertical_iou_merging_threshold': 0.5,
            'probability_random_scrolling': 0.5,
            'probability_change_background': probability_change_background,
            'probability_change_text_color_all_page': all_page_prob,
            'probability_change_font_all_page': all_page_prob,
            'probability_change_font_size_all_page': all_page_prob,
            'probability_change_text_decoration_all_page': all_page_prob,
            'probability_change_font_style_all_page': all_page_prob,
            'probability_change_font_weight_all_page': all_page_prob,
            'probability_change_font_variant_all_page': all_page_prob,
            'probability_change_font_stretch_all_page': all_page_prob,
            'probability_change_text_color': words_prob,
            'probability_change_font': words_prob,
            'probability_change_font_size': words_prob,
            'probability_change_text_decoration': words_prob,
            'probability_change_font_style': words_prob,
            'probability_change_font_variant': words_prob,
            'probability_change_font_weight': words_prob,
            'probability_change_font_stretch': words_prob,
            'force_direction': force_direction,
            'char_level': False,
            'rectangular_bounding_boxes': True,
            'detection_data_classes_field_names': ('bounding_boxes_text',),
            'detection_data_fields_to_mask_in_loss': ('bounding_boxes_images',),
            'consider_elastic': True,
            'consider_perspective': True,
            'enable_icon_or_emoji_text': False,
            'dictionary': None,
            'probability_replace_word_from_dictionary': 0.0,
            'generate_layout_data': True,
            'probability_change_by_web_modifier': 1.0,
            'webpage_modifier': modifier
        }
        generate_ocr_data(**kwargs)

