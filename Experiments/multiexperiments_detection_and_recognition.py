import json
import string
import os
from Config.setting import *
from OcrPipeline.ocr_training_pipeline import OcrTrainingPipeline
import matplotlib.pyplot as plt
from tabulate import tabulate
from GeneralUtils.project_logs import create_log
from Data.api import create_language


class MultiExperimentDetectionRecognition:
    def __init__(self, name, detection_model_names=tuple(), script_identification_model_names=tuple(), recognition_model_names=tuple()):
        self.name = name
        self.experiments_path = os.path.join(OCR_PIPELINES_TRAINING_PATH, 'Experiment_' + name)
        self.recognition_experiments_path = os.path.join(self.experiments_path, 'recognition')
        self.detection_experiments_path = os.path.join(self.experiments_path, 'detection')
        self.detection_model_names = list(detection_model_names)
        self.recognition_model_names = list(recognition_model_names)
        self.script_identification_model_names = list(script_identification_model_names)
        # self.script_identification_experiments_path = os.path.join(self.experiments_path, 'script_identification')
        self.running_info_dict_recognition = None
        self.running_info_dict_detection = None
        self.running_info_dict_script_identification = None
        os.makedirs(self.experiments_path, exist_ok=True)
        os.makedirs(self.recognition_experiments_path, exist_ok=True)
        os.makedirs(self.detection_experiments_path, exist_ok=True)
        # os.makedirs(self.script_identification_experiments_path, exist_ok=True)
        self.recognition_experiment_file = os.path.join(self.recognition_experiments_path, 'running_info.json')
        self.log = create_log(self.name, self.experiments_path)
        if os.path.isfile(self.recognition_experiment_file):
            self.log.info('Loading Recognition Experiment')
            self.running_info_dict_recognition = json.load(open(self.recognition_experiment_file, mode='r', encoding='utf-8'))
        self.detection_experiment_file = os.path.join(self.detection_experiments_path, 'running_info.json')
        if os.path.isfile(self.detection_experiment_file):
            self.log.info('Loading Detection Experiment')
            self.running_info_dict_detection = json.load(open(self.detection_experiment_file, mode='r', encoding='utf-8'))
        # self.script_identification_experiment_file = os.path.join(self.script_identification_experiments_path, 'running_info.json')
        # if os.path.isfile(self.script_identification_experiment_file):
        #     self.log.info('Loading Script Identification Experiment')
        #     self.running_info_dict_script_identification = json.load(open(self.script_identification_experiment_file, mode='r', encoding='utf-8'))

    def _save(self, running_info_dict, running_info_file):
        json_obj = json.dumps(running_info_dict, indent=4)
        with open(running_info_file, mode='w', encoding='utf-8') as fw:
            fw.write(json_obj)

    def _load(self, running_info_file):
        running_info_dict = json.load(open(running_info_file, mode='r', encoding='utf-8'))
        for k, v in running_info_dict.items():
            setattr(self, k, v)

    def _load_dict(self, running_info_dict):
        for k, v in running_info_dict.items():
            setattr(self, k, v)

    @staticmethod
    def _compare_validation_while_training_scores(loss_func_name, experiment_path, training_names_to_training_data, model_type='recognition'):
        training_names = list(training_names_to_training_data.keys())
        plt.figure(1, figsize=(24, 13), dpi=200)
        for training in training_names_to_training_data.keys():
            if model_type != 'recognition' or training.split('_')[-1] == loss_func_name:
                train_loss = training_names_to_training_data[training]['train_loss']
                plt.plot(train_loss, label=training)
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(experiment_path, 'TrainingLoss_{}.png'.format(loss_func_name)))
        plt.close()
        if model_type == 'detection':
            plt.figure(2, figsize=(24, 13), dpi=200)
            for training in training_names_to_training_data.keys():
                val_loss = training_names_to_training_data[training]['val_loss']
                plt.plot(val_loss, label=training)
            plt.title('Validation Loss')
            plt.legend()
            plt.savefig(os.path.join(experiment_path, 'compare_val_loss.png'))
            plt.close()
        else:
            for k in training_names_to_training_data[training_names[0]]['val_evaluation'][0].keys():
                plt.figure(2, figsize=(24, 13), dpi=200)
                for training_name in training_names:
                    if k != 'val_loss' or training_name.split('_')[-1] == loss_func_name:
                        num_epochs = len(training_names_to_training_data[training_name]['val_evaluation'])
                        plt.plot([training_names_to_training_data[training_name]['val_evaluation'][i][k] for i in range(num_epochs)], label=training_name)
                if k == 'val_loss':
                    printed_k = '{}_{}'.format(k, loss_func_name)
                else:
                    printed_k = k
                plt.title(f'{printed_k}')
                plt.legend()
                plt.savefig(os.path.join(experiment_path, f'compare_{printed_k}.png'))
                plt.close()

    @staticmethod
    def _compare_models(experiment_path, training_names_to_training_data):
        table_data = [['Model Name', 'Training Time', '#parameters']]
        for k in training_names_to_training_data.keys():
            table_data.append([k, training_names_to_training_data[k]['duration'],
                               training_names_to_training_data[k]['num_parameters']])
        table_data = tabulate(table_data, stralign='center')
        with open(os.path.join(experiment_path, 'models_comparison.txt'), mode='w',
                  encoding='utf-8') as f:
            f.write(table_data + '\n')

    @staticmethod
    def _compare_testing_scores(experiment_path, training_names_to_test_scores):
        model_names = [k for k in training_names_to_test_scores.keys()]
        test_sets = [k for k in training_names_to_test_scores[model_names[0]].keys()]
        test_names = [k for k in training_names_to_test_scores[model_names[0]][test_sets[0]].keys()]
        for test_set in test_sets:
            for test_name in test_names:
                compare = [[k for k in training_names_to_test_scores[model_names[0]][test_set][test_name].keys()]]
                for model_name in model_names:
                    compare.append([model_name] + [training_names_to_test_scores[model_name][test_set][test_name][k] for k in compare[0]])
                compare[0] = ['Model Name'] + compare[0]
                table_data = tabulate(compare, stralign='center', tablefmt='fancy_grid')
                with open(os.path.join(experiment_path, '{}_{}.txt'.format(test_name, test_set)), mode='w', encoding='utf-8') as f:
                    f.write(table_data + '\n')

    def running_recognition_training_configurations(self,
                                                    train_data_names=tuple(),
                                                    val_data_names=tuple(),
                                                    test_data_names=tuple(),
                                                    trained_detection_name=None,
                                                    charset=string.ascii_letters + string.digits + string.punctuation,
                                                    direction="ltr",
                                                    charset_to_charset_normalization=None,
                                                    language_detection=None,
                                                    language_detection_threshold=0.5,
                                                    recognition_batch_size=64,
                                                    recognition_num_epochs=150,
                                                    recognition_lr=1e-5,
                                                    recognition_gpu_ids='0',
                                                    recognition_hidden_size=512,
                                                    recognition_num_layers=4,
                                                    recognition_num_heads_self_attention=16,
                                                    recognition_dropout_self_attention=0.0,
                                                    recognition_max_word_length=1000,
                                                    recognition_image_size=(48, 150),
                                                    recognition_resize_method="ar",
                                                    pretrained_recognition_name=None,
                                                    num_workers=1,
                                                    enable_icon_or_emoji_text=False,
                                                    loss_functions=('nll', 'ctc'),
                                                    encoder_types=('transformer', 'bilstm'),
                                                    decoder_types=('transformer', 'fc'),
                                                    positional_enconding_types=("embedding", "const"),
                                                    random_transformation_probs=(0.25, 0.0),
                                                    init_options=(True, False),
                                                    early_stop_by_recognition=('val_loss', 'min')
                                                    ):
        if self.running_info_dict_recognition is None:
            self.running_info_dict_recognition = {
                'train_data_names': train_data_names,
                'val_data_names': val_data_names,
                'test_data_names': test_data_names,
                'trained_detection_name': trained_detection_name,
                'charset': charset,
                'direction': direction,
                'charset_to_charset_normalization': charset_to_charset_normalization,
                'language_detection': language_detection,
                'language_detection_threshold': language_detection_threshold,
                'recognition_batch_size': recognition_batch_size,
                'recognition_num_epochs': recognition_num_epochs,
                'recognition_lr': recognition_lr,
                'recognition_gpu_ids': recognition_gpu_ids,
                'recognition_hidden_size': recognition_hidden_size,
                'recognition_num_layers': recognition_num_layers,
                'recognition_num_heads_self_attention': recognition_num_heads_self_attention,
                'recognition_dropout_self_attention': recognition_dropout_self_attention,
                'recognition_max_word_length': recognition_max_word_length,
                'recognition_image_size': recognition_image_size,
                'recognition_resize_method': recognition_resize_method,
                'pretrained_recognition_name': pretrained_recognition_name,
                'num_workers': num_workers,
                'early_stop_by_recognition': early_stop_by_recognition,
                'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
                'loss_functions': loss_functions,
                'encoder_types': encoder_types,
                'decoder_types': decoder_types,
                'positional_enconding_types': positional_enconding_types,
                'init_options': init_options,
                'random_transformation_probs': random_transformation_probs,
                'trainings_output': {}
            }
            self._save(self.running_info_dict_recognition, self.recognition_experiment_file)
            self._load_dict(self.running_info_dict_recognition)
        else:
            self._load(self.recognition_experiment_file)
        for loss_func_name in self.loss_functions:
            for encoder_type in self.encoder_types:
                for decoder_type in self.decoder_types:
                    if encoder_type == 'transformer' or decoder_type == 'transformer':
                        positional_enconding_type_options = self.positional_enconding_types
                    else:
                        positional_enconding_type_options = ("const",)
                    for positional_enconding_type in positional_enconding_type_options:
                        for init_pos_as_const in self.init_options:
                            for random_transformation in self.random_transformation_probs:
                                conf = '_{}_{}_{}_{}_{}_{}'.format(encoder_type, decoder_type, positional_enconding_type,
                                                                'init' if init_pos_as_const else 'no_init', random_transformation, loss_func_name)
                                training_name = self.name + conf
                                if training_name not in self.running_info_dict_recognition['trainings_output'].keys():
                                    trainings_data = OcrTrainingPipeline(
                                        training_name,
                                        training_already_generated_dataset_names=self.train_data_names,
                                        val_already_generated_dataset_names=self.val_data_names,
                                        test_already_generated_dataset_names=self.test_data_names,
                                        charset=self.charset,
                                        direction=self.direction,
                                        charset_to_charset_normalization=self.charset_to_charset_normalization,
                                        language_detection=self.language_detection,
                                        language_detection_threshold=self.language_detection_threshold,
                                        generate_V1=False,
                                        generate_V2=False,
                                        style_transfer_training=False,
                                        augmenting_data=False,
                                        detection_training=False,
                                        recognition_training=True,
                                        recognition_batch_size=self.recognition_batch_size,
                                        recognition_num_epochs=self.recognition_num_epochs,
                                        recognition_lr=self.recognition_lr,
                                        recognition_gpu_ids=self.recognition_gpu_ids,
                                        recognition_random_transformation=random_transformation,
                                        recognition_max_word_length=self.recognition_max_word_length,
                                        recognition_encoder_type=encoder_type,
                                        recognition_hidden_size=self.recognition_hidden_size,
                                        recognition_num_layers=self.recognition_num_layers,
                                        recognition_num_heads_self_attention=self.recognition_num_heads_self_attention,
                                        recognition_dropout_self_attention=self.recognition_dropout_self_attention,
                                        recognition_without_rotated_data=False,
                                        pretrained_recognition_name=self.pretrained_recognition_name,
                                        recognition_image_size=self.recognition_image_size,
                                        recognition_resize_method=self.recognition_resize_method,
                                        recognition_loss_func_name=loss_func_name,
                                        recognition_decoder_type=decoder_type,
                                        recognition_positional_enconding_type=positional_enconding_type,
                                        recognition_init_pos_as_const=init_pos_as_const,
                                        num_workers_recognition=self.num_workers,
                                        pretrained_detection_name=self.trained_detection_name,
                                        prepare_recognition_data=False,
                                        compress_datasets_to_jpeg=False,
                                        enable_icon_or_emoji_text=self.enable_icon_or_emoji_text,
                                        early_stop_by_recognition=self.early_stop_by_recognition
                                    )
                                    self.running_info_dict_recognition['trainings_output'][training_name] = ({k: v for k, v in trainings_data.recognition_training_data.items() if 'state_dict' not in k}, trainings_data.tests_scores)
                            self._compare_validation_while_training_scores(loss_func_name, self.recognition_experiments_path, training_names_to_training_data={k: v[0] for k, v in self.running_info_dict_recognition['trainings_output'].items()}, model_type='recognition')
                            self._compare_models(self.recognition_experiments_path, training_names_to_training_data={k: v[0] for k, v in self.running_info_dict_recognition['trainings_output'].items()})
                            self._compare_testing_scores(self.recognition_experiments_path, training_names_to_test_scores={k: v[1] for k, v in self.running_info_dict_recognition['trainings_output'].items()})
                            self._save(self.running_info_dict_recognition, self.recognition_experiment_file)
        self.recognition_model_names = list(set(self.recognition_model_names + [k for k in self.running_info_dict_recognition['trainings_output'].keys()]))
        return self.running_info_dict_recognition

    def running_detection_training_configurations(self,
                                                  train_data_names=tuple(),
                                                  val_data_names=tuple(),
                                                  test_data_names=tuple(),
                                                  # charset=string.digits + string.ascii_letters + string.punctuation,
                                                  # direction="ltr",
                                                  detection_model_class="UNet",   #"Unet2D_Diffusers",
                                                  trained_recognition_name=None,
                                                  pretrained_detection_name=None,
                                                  detection_num_epochs=150,
                                                  detection_batch_size=8,
                                                  detection_batch_size_inference=8,
                                                  detection_lr=1e-4,
                                                  detection_gpu_ids='0',
                                                  num_workers=1,
                                                  inference_detection_post_processing=False,
                                                  inference_detection_post_processing_methods=('size_based_nms',),
                                                  inference_detection_post_processing_params=None,
                                                  enable_icon_or_emoji_text=True,
                                                  random_transformation_probs=(0.25, 0.0),
                                                  detection_gamma_scheduler=0.9,
                                                  detection_num_epochs_scheduler=10,
                                                  crop=(320, 320),
                                                  train_uda=False,
                                                  src_uda_folder='',
                                                  lambda_uda=1,
                                                  uda_train_only_discriminator=0,
                                                  accumulate=1
                                                  ):
        if self.running_info_dict_detection is None:
            self.running_info_dict_detection = {
                'train_data_names': train_data_names,
                'val_data_names': val_data_names,
                'test_data_names': test_data_names,
                # 'charset': charset,
                # 'direction': direction,
                'trained_recognition_name': trained_recognition_name,
                'pretrained_detection_name': pretrained_detection_name,
                'detection_num_epochs': detection_num_epochs,
                'detection_batch_size': detection_batch_size,
                'detection_batch_size_inference': detection_batch_size_inference,
                'detection_lr': detection_lr,
                'detection_gpu_ids': detection_gpu_ids,
                'num_workers': num_workers,
                'random_transformation_probs': random_transformation_probs,
                'detection_model_class': detection_model_class,
                'inference_detection_post_processing': inference_detection_post_processing,
                'inference_detection_post_processing_methods': inference_detection_post_processing_methods,
                'inference_detection_post_processing_params': inference_detection_post_processing_params,
                'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
                'detection_gamma_scheduler': detection_gamma_scheduler,
                'detection_num_epochs_scheduler': detection_num_epochs_scheduler,
                'crop': crop,
                'train_uda': train_uda,
                'src_uda_folder': src_uda_folder,
                'lambda_uda': lambda_uda,
                'uda_train_only_discriminator': uda_train_only_discriminator,
                'accumulate': accumulate,
                'trainings_output': {}
            }
            self._save(self.running_info_dict_detection, self.detection_experiment_file)
            self._load_dict(self.running_info_dict_detection)
        else:
            self._load(self.detection_experiment_file)
        for random_transformation in self.random_transformation_probs:
            conf = '_{}'.format(random_transformation)
            training_name = self.name + conf
            if training_name not in self.running_info_dict_detection['trainings_output'].keys():
                trainings_data = OcrTrainingPipeline(
                    training_name,
                    generate_V1=False,
                    generate_V2=False,
                    style_transfer_training=False,
                    augmenting_data=False,
                    detection_training=True,
                    recognition_training=False,
                    # charset=self.charset,
                    # direction=self.direction,
                    detection_num_epochs=self.detection_num_epochs,
                    training_already_generated_dataset_names=self.train_data_names,
                    val_already_generated_dataset_names=self.val_data_names,
                    test_already_generated_dataset_names=self.test_data_names,
                    pretrained_detection_name=self.pretrained_detection_name,
                    detection_batch_size=self.detection_batch_size,
                    detection_batch_size_inference=self.detection_batch_size_inference,
                    detection_gpu_ids=self.detection_gpu_ids,
                    num_workers_detection=self.num_workers,
                    detection_model_class=self.detection_model_class,
                    pretrained_recognition_name=self.trained_recognition_name,
                    detection_lr=self.detection_lr,
                    prepare_recognition_data=False,
                    compress_datasets_to_jpeg=False,
                    enable_icon_or_emoji_text=self.enable_icon_or_emoji_text,
                    inference_detection_post_processing=self.inference_detection_post_processing,
                    inference_detection_post_processing_methods=self.inference_detection_post_processing_methods,
                    inference_detection_post_processing_params=self.inference_detection_post_processing_params,
                    detection_random_transformation=random_transformation,
                    detection_gamma_scheduler=self.detection_gamma_scheduler,
                    detection_num_epochs_scheduler=self.detection_num_epochs_scheduler,
                    crop=self.crop,
                    train_uda=self.train_uda,
                    src_uda_folder=self.src_uda_folder,
                    lambda_uda=self.lambda_uda,
                    uda_train_only_discriminator=self.uda_train_only_discriminator,
                    detection_accumulate=self.accumulate
                )
            self.running_info_dict_detection['trainings_output'][training_name] = ({k: v for k, v in trainings_data.detection_training_data.items() if ('state_dict' not in k and 'classes_colors' not in k)}, trainings_data.tests_scores)
            self._compare_validation_while_training_scores('nll', self.detection_experiments_path, training_names_to_training_data={k: v[0] for k, v in self.running_info_dict_detection['trainings_output'].items()}, model_type='detection')
            self._compare_models(self.detection_experiments_path, training_names_to_training_data={k: v[0] for k, v in self.running_info_dict_detection['trainings_output'].items()})
            self._compare_testing_scores(self.detection_experiments_path, training_names_to_test_scores={k: v[1] for k, v in self.running_info_dict_detection['trainings_output'].items()})
            # self._compare_
            self._save(self.running_info_dict_detection, self.detection_experiment_file)
        self.detection_model_names = list(set(self.detection_model_names + [k for k in self.running_info_dict_detection['trainings_output'].keys()]))
        return self.running_info_dict_detection

