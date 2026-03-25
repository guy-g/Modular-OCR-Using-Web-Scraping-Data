import os
from PIL import Image
from selenium import webdriver
import numpy as np
import base64
import io
from tqdm import tqdm
import json
import shutil
from Data.dictionary_view import DictionaryView
from Data.fonts_view import FontsView
from Data.language import Language
from Data.Utils.string_representation import string_repr
from GeneralUtils.utils import *


COLOR_HEX_RANGE = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
TEXT_DECORATION_OPTIONS = ["none", "overline", "line-through", "underline", "underline overline"]
FONT_STYLE_OPTIONS = ["normal", "italic", "oblique"]
FONT_VARIANT_OPTIONS = ["normal", "small-caps"]
FONT_WEIGHT_OPTIONS = ["normal", "bold", "bolder", "lighter", 100, 200, 300, 400, 500, 600, 700, 800, 900]
FONT_FAMILY_OPTIONS = ["serif", "san-serif", "monospace", "cursive", "emoji", "math"]
FONT_STRETCH_OPTIONS = ["ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "normal", "semi-expanded", "expanded", "extra-expanded", "ultra-expanded"]
FONT_SIZE_RANGE = [11, 45]  #[5, 45]
LETTER_SPACING_RANGE = [1, 15]


def get_random_color():
    color = ['#'] + np.random.choice(COLOR_HEX_RANGE, 6).tolist()
    color = ''.join(color)
    return color


def build_html(html_dst_path, font_paths):
    html_str = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8" content="text/html">
'''
    for font_path in font_paths:
        html_str += '''
            <style>
                @font-face {{
                  font-family: {};
                  src: url(\"{}\");
                }}
            </style>
        '''.format(font_path.split(os.sep)[-1][:-4], font_path)
    html_str += '''
    </head>
    <body>
        <span style="margin: 300px 300px 300px 300px;">StartHere</span>
    </body>
</html>
'''
    with open(html_dst_path, mode='w', encoding='utf-8') as f:
        f.write(html_str)


def render_text(word_text, word_image_path, mask_image_path, font, direction, rotation_3d_probability, max_random_padding, driver):
    if np.random.rand() < rotation_3d_probability:
        rotation_code = 'element.style.transform = "rotate3d({}, {}, {}, {}deg)";'.format(
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-90, 90)
        )
        rotated = True
    else:
        rotation_code = ''
        rotated = False
    first_font_color = get_random_color()
    second_font_color = get_random_color()
    while first_font_color == second_font_color:
        first_font_color = get_random_color()
        second_font_color = get_random_color()

    main_js_code = '''
var element = document.querySelector("span");
element.innerText = "{}";
element.style.unicodeBidi = "bidi-override";
element.style.direction = "{}";
element.style.whiteSpace = "nowrap";
element.style.backgroundColor = "{}";
document.body.style.backgroundColor = element.style.backgroundColor;
element.style.webkitTextFillColor = "{}";
element.style.textDecoration = "{}";
element.style.fontStyle = "{}";
element.style.fontVariant = "{}";
element.style.fontWeight = "{}";
element.style.fontStretch = "{}";
element.style.fontFamily = "{}";
element.style.fontSize = "{}px";
element.style.letterSpacing = "{}px";
{}
element.style.display = "inline-block";
return [element, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.clientWidth, document.documentElement.scrollWidth];
    '''.format(
        string_repr(word_text),
        direction,
        get_random_color(),
        first_font_color,
        np.random.choice(["none", "none", "none", "none"] + TEXT_DECORATION_OPTIONS),
        np.random.choice(FONT_STYLE_OPTIONS),
        np.random.choice(FONT_VARIANT_OPTIONS),
        np.random.choice(FONT_WEIGHT_OPTIONS),
        np.random.choice(FONT_STRETCH_OPTIONS),
        font,
        np.random.randint(FONT_SIZE_RANGE[0], FONT_SIZE_RANGE[1]),
        np.random.randint(LETTER_SPACING_RANGE[0], LETTER_SPACING_RANGE[1] + 1),
        rotation_code
    )
    [element, h1, h2, w1, w2] = driver.execute_script(main_js_code)
    first_page_image = Image.open(io.BytesIO(base64.b64decode(driver.get_screenshot_as_base64()))).convert('RGB')
    driver.execute_script('''
var element = document.querySelector("span");
element.style.webkitTextFillColor = "{}";
    '''.format(second_font_color))
    second_page_image = Image.open(io.BytesIO(base64.b64decode(driver.get_screenshot_as_base64()))).convert('RGB')
    mask_image = Image.fromarray((np.abs(
        np.array(first_page_image) - np.array(second_page_image)).sum(axis=2) != 0).astype(
        np.uint8) * 255)
    y, x = np.where(np.array(mask_image) > 0)
    if len(y) > 0 and h1 == h2 and w1 == w2:
        element_bounding_box = [max(int(x.min()) - np.random.randint(0, max_random_padding + 1), 0),
                                max(int(y.min()) - np.random.randint(0, max_random_padding + 1), 0),
                                min(int(x.max()) + np.random.randint(0, max_random_padding + 1), mask_image.size[0] - 1),
                                min(int(y.max()) + np.random.randint(0, max_random_padding + 1), mask_image.size[1] - 1)]
        element_image = first_page_image.crop(element_bounding_box)
        element_image.save(word_image_path, 'PNG', compress_level=0)
        element_mask_image = mask_image.crop(element_bounding_box)
        element_mask_image.save(mask_image_path, 'PNG')  #, compress_level=0)
        return (element_image.height, element_image.width), rotated
    else:
        return None, None


def generate_recognition_by_words_and_fonts_lists(dst_folder, font_view, words, fonts, direction, rotation_3d_probability,
                                                  max_random_padding, chromedriver, log, window_size):
    shutil.rmtree(dst_folder, ignore_errors=True)
    os.makedirs(dst_folder, exist_ok=True)
    recognition_folder = os.path.join(dst_folder, 'recognition_data')
    os.makedirs(recognition_folder, exist_ok=True)
    json_recognition_file = os.path.join(recognition_folder, 'data.json')
    html_dst_path = os.path.join(recognition_folder, 'index.html')
    build_html(html_dst_path, font_view.font_paths)
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-web-security")
    options.add_argument("--headless")
    options.add_argument("window-size={},{}".format(window_size[0], window_size[1]))
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(executable_path=chromedriver, options=options)
    driver.get('file://' + os.path.abspath(html_dst_path))
    file_names_to_text = {}
    iword = 0
    while len(file_names_to_text.keys()) < len(words):
        file_name = str(iword) + '.png'
        mask_file_name = str(iword) + '_mask.png'
        image_size = None
        word = words[iword % len(words)]
        font = fonts[iword % len(words)]
        try:
            image_size, rotated = render_text(word, os.path.join(recognition_folder, file_name), os.path.join(recognition_folder, mask_file_name), font, direction,
                                 rotation_3d_probability, max_random_padding, driver)
        except Exception as e:
            log.exception(str(e))
            image_size = None
        if image_size is not None:
            file_names_to_text[file_name] = {
                        'text': word,
                        'image_size': image_size,
                        'mask_file': os.path.join(recognition_folder, mask_file_name),
                        'rotated': rotated,
                        'direction': direction
                    }
        iword += 1
        if len(file_names_to_text.keys()) % 10000 == 0:
            log.info(str(len(file_names_to_text.keys())))
            save_to_json(file_names_to_text, json_recognition_file)
    log.info(str(len(file_names_to_text.keys())))
    save_to_json(file_names_to_text, json_recognition_file)
    driver.close()
    driver.quit()
    save_to_json(file_names_to_text, json_recognition_file)
    os.remove(html_dst_path)


def generate_recognition(dst_folder, running_name, log, dictionary_view: DictionaryView, font_view: FontsView,
                         language: Language, num_samples: int, samples_by_order=False, rotation_3d_probability=0.5,
                         max_random_padding=5, chromedriver: str = r'Data/necessary_files/chromedriver', window_size=(1920, 1080)):
    os.makedirs(dst_folder, exist_ok=True)
    running_folder = os.path.join(dst_folder, running_name)
    words = []
    fonts = []
    log.info('Choosing texts and fonts')
    for i in tqdm(range(num_samples)):
        words.append(dictionary_view.get_word(samples_by_order))
        fonts.append(font_view.get_font(without_extansion=True))
    generate_recognition_by_words_and_fonts_lists(running_folder, font_view, words, fonts, language.direction,
                                                  rotation_3d_probability, max_random_padding, chromedriver, log, window_size)
    running_info = {
        'dst_folder': dst_folder,
        'running_name': running_name,
        'dictionary_view_name': dictionary_view.dictionary_view_name,
        'fonts_view_name': font_view.fonts_view_name,
        'num_samples': num_samples,
        'chromedriver': chromedriver,
        'direction': str(language.direction)
    }
    save_to_json(running_info, os.path.join(running_folder, 'running_info.json'))

