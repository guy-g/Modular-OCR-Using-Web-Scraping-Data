import numpy as np

from Detection.api import *
from Recognition.api import *
from ScriptIdentification.api import *
from Preprocessing.preprocess import preprocess_image
from GeneralUtils.draw_demonstration import draw_images_paths_to_bounding_boxes_and_text
from Chunking.chunking import Chunker
from PIL import Image


class OcrInferencePipeline:
    def __init__(self,
                 pipeline_name,
                 font_file=None,
                 detection_name=None,
                 recognition_names=(None,),
                 script_identification_name=None,
                 load_pipeline_file=False,
                 detection_batch_size=16,
                 crop=None,
                 detection_gpu_ids='0',
                 detection_num_workers=1,
                 detection_padding=1,
                 detection_binary_mask_threshold=0.95,
                 detection_post_processing=False,
                 detection_post_processing_methods=('size_based_nms',),
                 detection_post_processing_params=None,
                 recognition_batch_size=16,
                 recognitions_gpu_ids=('0',),
                 recognition_num_workers=1,
                 recognition_resize_method=None,
                 whole_image_preprocessing=None,
                 every_word_preprocessing=None,
                 script_identification_gpu_ids='0',
                 script_identification_num_workers=1,
                 script_identification_binary_threshold=0.5,
                 script_identification_decision_method='first',
                 script_identification_priority=None,
                 recognition_chunking_params=None,
                 script_identification_chunking_params=None,
                 use_onnx=False,
                 uda_data=False,
                 max_image_shape=2000
                 ):
        self.pipeline_file = os.path.join(OCR_PIPELINES_INFERENCE_PATH, pipeline_name + '.json')
        clear_cache()
        if not load_pipeline_file:
            self.pipeline_name = pipeline_name
            self.use_onnx = use_onnx
            self.model_ext = '.onnx' if self.use_onnx else '.pt'
            self.font_file = font_file
            self.detection_batch_size = detection_batch_size
            self.crop = crop
            self.detection_gpu_ids = detection_gpu_ids
            self.detection_padding = detection_padding
            self.detection_binary_mask_threshold = detection_binary_mask_threshold
            self.recognition_batch_size = recognition_batch_size
            self.recognitions_gpu_ids = list(recognitions_gpu_ids)
            self.recognition_resize_method = recognition_resize_method
            self.whole_image_preprocessing = whole_image_preprocessing
            self.every_word_preprocessing = every_word_preprocessing
            self.detection_post_processing = detection_post_processing
            self.detection_post_processing_methods = detection_post_processing_methods
            self.detection_post_processing_params = detection_post_processing_params
            self.script_identification_gpu_ids = script_identification_gpu_ids
            self.script_identification_binary_threshold = script_identification_binary_threshold
            self.detection_num_workers = detection_num_workers
            self.recognition_num_workers = recognition_num_workers
            self.script_identification_num_workers = script_identification_num_workers
            self.detection_name = detection_name
            self.recognition_names = list(recognition_names)
            self.script_identification_name = script_identification_name
            self.recognition_chunking_params = recognition_chunking_params
            self.script_identification_chunking_params = script_identification_chunking_params
            self.script_identification_decision_method = script_identification_decision_method
            self.script_identification_priority = script_identification_priority
            self.uda_data = uda_data
            self.max_image_shape = max_image_shape
            self.save()
        else:
            self.__load()
        setattr(self, 'detection_name', self.detection_name)
        setattr(self, 'recognition_names', self.recognition_names)
        setattr(self, 'script_identification_name', self.script_identification_name)
        setattr(self, 'recognition_chunking_params', self.recognition_chunking_params)
        setattr(self, 'script_identification_chunking_params', self.script_identification_chunking_params)

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key == 'detection_name':
            if self.detection_name is not None:
                self.detection_model = load_model_by_gpu_ids(os.path.join(DETECTION_MODELS_PATH, self.detection_name, 'segmentation' + self.model_ext), self.detection_gpu_ids)
                self.detection_data = None
            else:
                self.detection_model = None
                self.detection_data = None
        elif key == 'recognition_names':
            self.recognition_models = []
            self.recognition_datas = []
            for irec, recognition_name in enumerate(self.recognition_names):
                if recognition_name is not None:
                    self.recognition_models.append(load_model_by_gpu_ids(os.path.join(RECOGNITION_MODELS_PATH, recognition_name, 'recognition' + self.model_ext), self.recognitions_gpu_ids[irec]))
                    self.recognition_datas.append(torch.load(os.path.join(RECOGNITION_MODELS_PATH, recognition_name, 'running_data.pt'), map_location='cpu'))
                else:
                    self.recognition_models.append(None)
                    self.recognition_datas.append(None)
        elif key == 'script_identification_name':
            if self.script_identification_name is not None:
                self.script_identification_model = load_model_by_gpu_ids(os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, self.script_identification_name, 'script_identification' + self.model_ext),
                                                                         self.script_identification_gpu_ids)
                self.script_identification_data = torch.load(os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, self.script_identification_name, 'running_data.pt'), map_location='cpu')
                # assert set(self.script_identification_model.scripts) == set(recognition_names)
            else:
                self.script_identification_model = None
                self.script_identification_data = None
        elif key == 'recognition_chunking_params':
            if self.recognition_chunking_params is not None:
                self.chunker_recognition = Chunker(**self.recognition_chunking_params)
        elif key == 'script_identification_chunking_params':
            if self.script_identification_chunking_params is not None:
                self.chunker_script_identification = Chunker(**self.script_identification_chunking_params)

    def save(self):
        data = {
            'pipeline_name': self.pipeline_name,
            'font_file': self.font_file,
            'detection_name': self.detection_name,
            'recognition_names': self.recognition_names,
            'script_identification_name': self.script_identification_name,
            'pipeline_file': self.pipeline_file,
            'detection_batch_size': self.detection_batch_size,
            'crop': self.crop,
            'detection_gpu_ids': self.detection_gpu_ids,
            'detection_padding': self.detection_padding,
            'detection_binary_mask_threshold': self.detection_binary_mask_threshold,
            'detection_post_processing': self.detection_post_processing,
            'detection_post_processing_methods': self.detection_post_processing_methods,
            'detection_post_processing_params': self.detection_post_processing_params,
            'recognition_batch_size': self.recognition_batch_size,
            'recognitions_gpu_ids': self.recognitions_gpu_ids,
            'recognition_resize_method': self.recognition_resize_method,
            'whole_image_preprocessing': self.whole_image_preprocessing,
            'every_word_preprocessing': self.every_word_preprocessing,
            'script_identification_gpu_ids': self.script_identification_gpu_ids,
            'script_identification_binary_threshold': self.script_identification_binary_threshold,
            'script_identification_num_workers': self.script_identification_num_workers,
            'script_identification_decision_method': self.script_identification_decision_method,
            'script_identification_priority': self.script_identification_priority,
            'recognition_num_workers': self.recognition_num_workers,
            'detection_num_workers': self.detection_num_workers,
            'recognition_chunking_params': self.recognition_chunking_params,
            'script_identification_chunking_params': self.script_identification_chunking_params,
            'use_onnx': self.use_onnx,
            'uda_data': self.uda_data,
            'max_image_shape': self.max_image_shape
        }
        json_object = json.dumps(data, indent=4)
        with open(self.pipeline_file, mode='w', encoding='utf-8') as data_file:
            data_file.write(json_object)

    def __load(self):
        data = json.load(open(self.pipeline_file, mode='r', encoding='utf-8'))
        self.pipeline_name = data['pipeline_name']
        self.use_onnx = data['use_onnx']
        self.model_ext = '.onnx' if self.use_onnx else '.pt'
        self.font_file = data['font_file']
        self.pipeline_file = data['pipeline_file']
        self.detection_batch_size = data['detection_batch_size']
        self.crop = data['crop']
        self.detection_gpu_ids = data['detection_gpu_ids']
        self.detection_padding = data['detection_padding']
        self.detection_binary_mask_threshold = data['detection_binary_mask_threshold']
        self.detection_post_processing = data['detection_post_processing']
        self.detection_post_processing_methods = data['detection_post_processing_methods']
        self.detection_post_processing_params = data['detection_post_processing_params']
        self.recognition_batch_size = data['recognition_batch_size']
        self.recognitions_gpu_ids = data['recognitions_gpu_ids']
        self.recognition_resize_method = data['recognition_resize_method']
        self.whole_image_preprocessing = data['whole_image_preprocessing']
        self.every_word_preprocessing = data['every_word_preprocessing']
        self.script_identification_gpu_ids = data['script_identification_gpu_ids']
        self.script_identification_binary_threshold = data['script_identification_binary_threshold']
        self.recognition_num_workers = data['recognition_num_workers']
        self.detection_num_workers = data['detection_num_workers']
        self.script_identification_num_workers = data['script_identification_num_workers']
        self.detection_name = data['detection_name']
        self.recognition_names = data['recognition_names']
        self.script_identification_name = data['script_identification_name']
        self.recognition_chunking_params = data['recognition_chunking_params']
        self.script_identification_chunking_params = data['script_identification_chunking_params']
        self.script_identification_decision_method = data['script_identification_decision_method']
        self.script_identification_priority = data['script_identification_priority']
        self.uda_data = data['uda_data']
        self.max_image_shape = data['max_image_shape']

    def inference_whole_images(self, inference_folder):
        words_folder = make_tmp_folder()
        image_paths_to_lines_and_words = self.inference_detection(inference_folder)
        for image_path, lines_and_words in image_paths_to_lines_and_words.items():
            shutil.rmtree(words_folder, ignore_errors=True)
            os.makedirs(words_folder, exist_ok=True)
            line_idxs_to_sub_image_paths = self.crop_image_by_bounding_boxes(image_path, lines_and_words, words_folder)
            sub_image_paths_to_text_and_scripts = self.inference_script_identification_and_recognition(words_folder)
            for line_idx, sub_image_paths in line_idxs_to_sub_image_paths.items():
                line_text = []
                for word_idx, sub_image_path in enumerate(sub_image_paths):
                    image_paths_to_lines_and_words[image_path][line_idx]['words'][word_idx]['text'] = sub_image_paths_to_text_and_scripts[sub_image_path]['text']
                    image_paths_to_lines_and_words[image_path][line_idx]['words'][word_idx]['script'] = sub_image_paths_to_text_and_scripts[sub_image_path]['script']
                    if self.uda_data:
                        line_text.append(sub_image_paths_to_text_and_scripts[sub_image_path]['text']['text'])
                    else:
                        line_text.append(sub_image_paths_to_text_and_scripts[sub_image_path]['text'])
                image_paths_to_lines_and_words[image_path][line_idx]['line_text'] = ' '.join(line_text)
        # if self.font_file is not None:
        #     draw_images_paths_to_bounding_boxes_and_text(images_paths_to_bounding_boxes_and_text, self.font_file)
        shutil.rmtree(words_folder, ignore_errors=True)
        return image_paths_to_lines_and_words

    def inference_script_identification_and_recognition(self, inference_folder):
        image_paths_to_text = {}
        image_paths_to_text_and_scripts = {}
        for idx, recognition_name in enumerate(self.recognition_names):
            image_paths_to_text[recognition_name] = self.inference_recognition(inference_folder, recognition_name)
        if len(self.recognition_names) > 1 and self.script_identification_model is not None:
            sub_image_paths_to_scripts = self.inference_script_identification(inference_folder)
            script_names = self.script_identification_data['language_names']
        else:
            sub_image_paths_to_scripts = {p: {recognition_name: 1 for recognition_name in self.recognition_names} for p in image_paths_to_text[self.recognition_names[0]].keys()}
            script_names = self.recognition_names
        for image_path in image_paths_to_text[self.recognition_names[0]].keys():
            image_script = [sub_image_paths_to_scripts[image_path][script_name] for script_name in script_names]
            image_script = self.recognition_names[image_script.index(max(image_script))]
            # image_paths_to_text[image_path] = image_paths_to_text[image_script][image_path]
            image_paths_to_text_and_scripts[image_path] = {'text': image_paths_to_text[image_script][image_path], 'script': image_script}
        for idx, recognition_name in enumerate(self.recognition_names):
            del image_paths_to_text[recognition_name]
        return image_paths_to_text_and_scripts

    def inference_recognition(self, inference_folder, recognition_name=None):
        recognition_folder = make_tmp_folder()
        for file_name in os.listdir(inference_folder):
            if file_name.lower().endswith(OCRDatasetView.IMAGES_EXT):
                image = Image.open(os.path.join(inference_folder, file_name)).convert('RGB')
                image = Image.fromarray(preprocess_image(np.array(image), self.every_word_preprocessing))
                image.save(os.path.join(recognition_folder, file_name + '.png'), compress_level=0)
        if recognition_name is None:
            recognition_name = self.recognition_names[0]
        recognition_gpu_ids = self.recognitions_gpu_ids[self.recognition_names.index(recognition_name)]
        recognition_model = self.recognition_models[self.recognition_names.index(recognition_name)]
        recognition_data = self.recognition_datas[self.recognition_names.index(recognition_name)]
        if self.recognition_chunking_params is not None and not self.uda_data:
            self.chunker_recognition.divide(recognition_folder)
        image_paths_to_text = inference_recognition(recognition_folder, recognition_name,
                                                        self.recognition_batch_size,
                                                        recognition_gpu_ids, model=recognition_model,
                                                        model_data=recognition_data,
                                                        num_workers=self.recognition_num_workers,
                                                        recognition_resize_method=self.recognition_resize_method,
                                                        uda_data=self.uda_data
                                                    )
        if self.recognition_chunking_params is not None and not self.uda_data:
            image_paths_to_text = self.chunker_recognition.merge(image_paths_to_text)
        image_paths_to_text = {os.path.join(inference_folder, k.split(os.sep)[-1][:-4]): v for k, v in image_paths_to_text.items()}
        shutil.rmtree(recognition_folder, ignore_errors=True)
        return image_paths_to_text

    def inference_script_identification(self, inference_folder):
        script_identification_folder = make_tmp_folder()
        for file_name in os.listdir(inference_folder):
            if file_name.lower().endswith(OCRDatasetView.IMAGES_EXT):
                image = Image.open(os.path.join(inference_folder, file_name)).convert('RGB')
                image = Image.fromarray(preprocess_image(np.array(image), self.every_word_preprocessing))
                image.save(os.path.join(script_identification_folder, file_name + '.png'), compress_level=0)
        if self.script_identification_chunking_params is not None and not self.uda_data:
            self.chunker_script_identification.divide(script_identification_folder)
        image_paths_to_scripts = inference_script_identification(script_identification_folder,
                                                                 self.script_identification_name,
                                                                 self.recognition_batch_size,
                                                                 self.script_identification_gpu_ids,
                                                                 self.script_identification_model,
                                                                 self.script_identification_data,
                                                                 self.script_identification_binary_threshold,
                                                                 self.script_identification_num_workers)
        if self.script_identification_chunking_params is not None and not self.uda_data:
            image_paths_to_scripts = self.chunker_script_identification.merge(image_paths_to_scripts)
        image_paths_to_scripts = {os.path.join(inference_folder, k.split(os.sep)[-1][:-4]): v for k, v in image_paths_to_scripts.items()}
        image_paths_to_scripts = self._script_identification_model_predictions_to_final_script_decisions(image_paths_to_scripts)
        shutil.rmtree(script_identification_folder, ignore_errors=True)
        return image_paths_to_scripts

    def inference_detection(self, inference_folder):
        detection_folder = make_tmp_folder()
        image_paths_to_scaling_factor = {}
        for file_name in os.listdir(inference_folder):
            if file_name.lower().endswith(OCRDatasetView.IMAGES_EXT):
                image = Image.open(os.path.join(inference_folder, file_name)).convert('RGB')
                image = Image.fromarray(preprocess_image(np.array(image), self.whole_image_preprocessing))
                image_paths_to_scaling_factor[os.path.join(inference_folder, file_name)] = 1
                while self.max_image_shape is not None and max(image.size) >= self.max_image_shape:
                    image = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
                    image_paths_to_scaling_factor[os.path.join(inference_folder, file_name)] *= 2
                image.save(os.path.join(detection_folder, file_name + '.png'), compress_level=0)
        image_paths_to_lines_and_words = inference_detection(detection_folder,
                                                             self.detection_name,
                                                             batch_size=self.detection_batch_size,
                                                             padding=self.detection_padding,
                                                             binary_mask_threshold=self.detection_binary_mask_threshold,
                                                             gpu_ids=self.detection_gpu_ids,
                                                             model=self.detection_model,
                                                             model_data=self.detection_data,
                                                             post_processing=self.detection_post_processing,
                                                             post_processing_methods=self.detection_post_processing_methods,
                                                             post_processing_params=self.detection_post_processing_params,
                                                             num_workers=self.detection_num_workers)
        image_paths_to_lines_and_words = {os.path.join(inference_folder, k.split(os.sep)[-1][:-4]): v for k, v in image_paths_to_lines_and_words.items()}
        for image_path in image_paths_to_lines_and_words.keys():
            for i in range(len(image_paths_to_lines_and_words[image_path])):
                scaling_factor = image_paths_to_scaling_factor[image_path]
                image_paths_to_lines_and_words[image_path][i]['line_bounding_box'] = [scaling_factor * item for item in image_paths_to_lines_and_words[image_path][i]['line_bounding_box']]
                for j in range(len(image_paths_to_lines_and_words[image_path][i]['words'])):
                    image_paths_to_lines_and_words[image_path][i]['words'][j]['bounding_box'] = [scaling_factor * item for item in image_paths_to_lines_and_words[image_path][i]['words'][j]['bounding_box']]
        shutil.rmtree(detection_folder, ignore_errors=True)
        return image_paths_to_lines_and_words

    def crop_image_by_bounding_boxes(self, image_path, lines_and_words, results_folder):
        line_idxs_to_sub_image_paths = {}
        image = Image.open(image_path).convert('RGB')
        count_sub_images = 0
        for iline, line in enumerate(lines_and_words):
            line_idxs_to_sub_image_paths[iline] = []
            for word in line['words']:
                bb_word = word['bounding_box']
                if bb_word[0] < bb_word[2] and bb_word[1] < bb_word[3]:
                    sub_image = image.crop(bb_word)
                    sub_image_path = os.path.join(results_folder, str(count_sub_images) + '.png')
                    sub_image.save(sub_image_path, compress_level=0)
                    line_idxs_to_sub_image_paths[iline].append(sub_image_path)
                    count_sub_images += 1
        return line_idxs_to_sub_image_paths

    # script_identification_decision_method = 'first', scripts_priority = None
    def _script_identification_model_predictions_to_final_script_decisions(self, image_paths_to_scripts):
        '''
        In general, word can be belonged to multiple scripts.
        Thus, our algorithm can return multiple scripts for each word image as a prediction, where for each
        script the algorithm use a separate binary classification header.
        Because we want only one string output for that word, we must continue with only one script.
        This is the purpose of this function
        '''
        image_paths_to_scripts_final = {}
        for image_path in image_paths_to_scripts.keys():
            if self.script_identification_decision_method == 'first':
                image_scripts = [image_paths_to_scripts[image_path][script_name] for script_name in self.script_identification_data['language_names']]
                image_script_idx = image_scripts.index(max(image_scripts))
                image_paths_to_scripts_final[image_path] = {script_name: 1 * (ik == image_script_idx) for ik, script_name in enumerate(self.script_identification_data['language_names'])}
            elif self.script_identification_decision_method == 'priority':
                for k in self.script_identification_priority:
                    if image_paths_to_scripts[image_path][k]:
                        image_paths_to_scripts_final[image_path] = {script_name: 1 * (k == script_name) for script_name in self.script_identification_data['language_names']}
                        break
        return image_paths_to_scripts_final

    @staticmethod
    def ocr_res_to_page(ocr_res):
        new_ocr_res = {}
        for k, v in ocr_res.items():
            lines_sorted = sorted(v, key=lambda x: (x['line_bounding_box'][1] + x['line_bounding_box'][3]) / 2.0)
            page = '\n'.join([line['line_text'] for line in lines_sorted])
            new_ocr_res[k] = page
        return new_ocr_res
