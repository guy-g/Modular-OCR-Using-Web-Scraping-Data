from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np


def draw_images_paths_to_bounding_boxes_and_text(images_paths_to_bounding_boxes_and_text, font_file, with_initial=True, images_paths_to_bounding_boxes_images=None):
    '''
    Note: demonstration doesn't need to be PNG without any compression, so it is always a compressed PNG.
    Further compression by JPEG is optional by the user.
    '''
    for image_path, bb_and_text_list in images_paths_to_bounding_boxes_and_text.items():
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, 12, encoding='utf-8')
        for (text, bb) in bb_and_text_list:
            if type(bb) == dict:
                bb_list = [bb['left'], bb['top'], bb['right'], bb['bottom']]
            else:
                bb_list = bb
            draw.rectangle([(bb_list[0], bb_list[1]), (bb_list[2], bb_list[3])], outline='red')
            draw.text((bb_list[0], bb_list[1] - 10), text, font=font, fill=(255, 0, 0))
        if images_paths_to_bounding_boxes_images is not None:
            for bb in images_paths_to_bounding_boxes_images[image_path]:
                if type(bb) == dict:
                    bb_list = [bb['left'], bb['top'], bb['right'], bb['bottom']]
                else:
                    bb_list = bb
                draw.rectangle([(bb_list[0], bb_list[1]), (bb_list[2], bb_list[3])], outline='red')
        if with_initial:
            image.save(image_path + '_demonstration.png')  #, compress_level=0)
        else:
            image.save(os.path.join('{}'.format(os.sep).join(image_path.split(os.sep)[:-1]), 'demonstration.png'))   #, compress_level=0)


def draw_images_paths_to_ocr_res(images_paths_to_ocr_res, font_file):
    '''
    Note: demonstration doesn't need to be PNG without any compression, so it is always a compressed PNG.
    Further compression by JPEG is optional by the user.
    '''
    for image_path, lines in images_paths_to_ocr_res.items():
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, 12, encoding='utf-8')
        for line in lines:
            # bb = line['line_bounding_box']
            # text = line['line_text']
            for word in line['words']:
                bb = word['bounding_box']
                text = word['text']
                if type(bb) == dict:
                    bb_list = [bb['left'], bb['top'], bb['right'], bb['bottom']]
                else:
                    bb_list = bb
                c = 'red'
                draw.rectangle([(bb_list[0], bb_list[1]), (bb_list[2], bb_list[3])], outline=c)
                draw.text((bb_list[0], bb_list[1] - 10), text, font=font, fill=c)

            # draw.rectangle([(bb_list[0], bb_list[1]), (bb_list[2], bb_list[3])], outline='red')
            # draw.text((bb_list[0], bb_list[1] - 10), text, font=font, fill=(255, 0, 0))
        image.save(image_path + '_demonstration.png')  #, compress_level=0)


def draw_images_paths_to_ocr_res_conf(images_paths_to_ocr_res, font_file):
    for image_path, lines in images_paths_to_ocr_res.items():
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, 12, encoding='utf-8')
        for line in lines:
            for word in line['words']:
                bb = word['bounding_box']
                if type(bb) == dict:
                    bb_list = [bb['left'], bb['top'], bb['right'], bb['bottom']]
                else:
                    bb_list = bb
                text = np.mean(word['text']['scores'])
                if text < 0.5:
                    c = 'red'
                else:
                    c = 'blue'
                text = '{:.2f}'.format(text)
                draw.rectangle([(bb_list[0], bb_list[1]), (bb_list[2], bb_list[3])], outline=c)
                draw.text((bb_list[0], bb_list[1] - 10), text, font=font, fill=c)
        image.save(image_path + '_demonstration.png')


def draw_images_paths_to_bounding_boxes(images_paths_to_bounding_boxes):
    for image_path, bbs in images_paths_to_bounding_boxes.items():
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        for bb in bbs:
            draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], outline='red', width=2)
        image.save(image_path + '_demonstration.png', compress_level=0)

