import os
import uuid
from Config.setting import *
from ScriptIdentification.script_identification import *
from GeneralUtils.project_exceptions import *
from GeneralUtils.project_logs import create_log


def train_script_identification(running_name, train_datasets_names, val_datasets_names,
                    language_names, random_transformation=0.25, num_epochs=100, batch_size=256, lr=1e-5, gpu_ids='0',
                    pretrained_script_name=None, log=None, image_size=(48, 150),
                    resize_method="ar", already_normalized=False, enable_icon_or_emoji_text=False, num_workers=1):
    os.makedirs(os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, running_name), exist_ok=True)
    if log is None:
        log = create_log(running_name, os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, running_name))
    try:
        running_data = train(running_name, train_datasets_names, val_datasets_names, language_names, LANGUAGES_PATH,
          DATASETS_VIEWS_PATH, DATASETS_PATH, SCRIPT_IDENTIFICATION_MODELS_PATH,
          log, num_epochs=num_epochs, batch_size=batch_size, lr=lr, random_transformation=random_transformation, gpu_ids=gpu_ids,
          pretrained_script_name=pretrained_script_name, image_size=image_size, resize_method=resize_method, already_normalized=already_normalized,
          enable_icon_or_emoji_text=enable_icon_or_emoji_text, num_workers=num_workers)
        log.info('Script identification training finished!')
        return running_data
    except Exception as e:
        log.exception(str(e))
        raise e


def inference_script_identification(inference_folder, training_name, batch_size=16, gpu_ids='0', model=None, model_data=None, binary_threshold=0.5, num_workers=1):
    if not os.path.isdir(os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, training_name)):
        raise ProjectIsNotExists()
    if model is None:
        model = os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, training_name, 'script_identification.pt')
    if model_data is None:
        model_data = torch.load(os.path.join(SCRIPT_IDENTIFICATION_MODELS_PATH, training_name, 'running_data.pt'), map_location='cpu')
    image_paths_to_scripts = inference(inference_folder, model, model_data, binary_threshold, batch_size, gpu_ids, num_workers)
    return image_paths_to_scripts

