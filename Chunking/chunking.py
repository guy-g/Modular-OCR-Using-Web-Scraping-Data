import os
import numpy as np
from PIL import Image
from uuid import uuid4
from Data.dataset_view import OCRDatasetView
from Data.SyntheticData.euclidean_utils import bounding_box_list_to_dict


class Chunker:
    def __init__(self, h_to_divide='*', w_to_divide=150, ar_to_divide=8, divide_method='divide_by_ar', divide_args=(150, 0), merge_method='append_text', merge_args=None):
        self.h_to_divide = h_to_divide if h_to_divide != '*' else np.inf
        self.w_to_divide = w_to_divide if w_to_divide != '*' else np.inf
        self.ar_to_divide = ar_to_divide if ar_to_divide != '*' else np.inf
        self.divide_method = divide_method
        self.divide_args = divide_args
        self.merge_method = merge_method
        self.merge_args = merge_args
        self._memory = {}
        self._created_images = []

    def _clear(self):
        self._memory = {}
        for image_path in self._created_images:
            os.remove(image_path)
        self._created_images = []

    def _need_division(self, np_image):
        image_size = (np_image.shape[0], np_image.shape[1])
        return self.chunker_need_to_split(image_size[0], image_size[1])

    def chunker_need_to_split(self, h, w):
        return (h > self.h_to_divide or w > self.w_to_divide) and ((float(w) / float(h)) >= self.ar_to_divide)

    def _image_to_division_bounding_boxes(self, np_image):
        if self.divide_method == 'divide_by_ar':
            bounding_boxes = self._divide_by_ar(np_image, *self.divide_args)
        return bounding_boxes

    def _divide_by_ar(self, np_image, win_size, win_overlap):
        image_size = (np_image.shape[0], np_image.shape[1])
        i = 0
        bounding_boxes = []
        while i < image_size[1]:
            bb = bounding_box_list_to_dict([i, 0, min(i + win_size - 1, image_size[1] - 1), image_size[0] - 1])
            bounding_boxes.append(bb)
            i = i + win_size - win_overlap
        return bounding_boxes

    def _divide_image(self, image_path):
        np_image = np.array(Image.open(image_path).convert('RGB'))
        self._memory[image_path] = []
        if self._need_division(np_image):
            bounding_boxes = self._image_to_division_bounding_boxes(np_image)
            for bb in bounding_boxes:
                sub_image = np_image[bb['top']: bb['bottom'] + 1, bb['left']: bb['right'] + 1, :]
                sub_image_path = image_path + str(uuid4()) + '.png'
                Image.fromarray(sub_image).save(sub_image_path, compress_level=0)
                self._memory[image_path].append({'sub_image_path': sub_image_path, 'sub_image_bounding_box': bb})
                self._created_images.append(sub_image_path)
        else:
            self._memory[image_path] = [{'sub_image_path': image_path, 'sub_image_bounding_box': None}]

    def _merge_results(self, results_and_bbs):
        if self.merge_method == 'append_text':
            res = self._append_text(results_and_bbs)
        elif self.merge_method == 'scores_sum':
            res = self._scores_sum(results_and_bbs)
        return res

    def _append_text(self, texts_and_bbs):
        word = ''
        for (t, bb) in texts_and_bbs:
            word += t
        return word

    def _scores_sum(self, results_and_bbs):
        scores_sum = {k: [] for k in results_and_bbs[0][0].keys()}
        for (r, bb) in results_and_bbs:
            for k in r.keys():
                scores_sum[k].append(r[k])
        scores_sum = {k: sum(v) for k, v in scores_sum.items()}
        return scores_sum

    def divide(self, inference_folder):
        self._clear()
        for fn in os.listdir(inference_folder):
            if fn.lower().endswith(OCRDatasetView.IMAGES_EXT):
                self._divide_image(os.path.join(inference_folder, fn))

    def merge(self, image_paths_to_results):
        src_image_paths_to_results = {}
        for src_image_path in self._memory.keys():
            sub_images_results_and_bbs = []
            for sub_image in self._memory[src_image_path]:
                sub_image_path = sub_image['sub_image_path']
                sub_image_bounding_box = sub_image['sub_image_bounding_box']
                sub_image_result = image_paths_to_results[sub_image_path]
                sub_images_results_and_bbs.append((sub_image_result, sub_image_bounding_box))
            res = self._merge_results(sub_images_results_and_bbs)
            src_image_paths_to_results[src_image_path] = res
        self._clear()
        return src_image_paths_to_results

