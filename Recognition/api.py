import os
import uuid
from Config.setting import *
from Recognition.recognition import *
from GeneralUtils.project_exceptions import *
from GeneralUtils.project_logs import create_log


def train_recognition(running_name, train_datasets_names, val_datasets_names,
                    language_name, random_transformation=0.25, max_word_length=1000, num_epochs=100, batch_size=256, lr=1e-5, loss_func_name='nll', gpu_ids='0',
                    encoder_type='bilstm', hidden_size=256, num_layers=4, num_heads_self_attention=4,
                    dropout_self_attention=0.0, without_rotated_data=False, pretrained_recognition_name=None, log=None, image_size=(48, 150),
                    recognition_resize_method="ar", already_normalized=False, decoder_type='fc', positional_enconding_type='const', init_pos_as_const=True,
                    enable_icon_or_emoji_text=False, num_workers=1, early_stop_by=('val_loss', 'min')):
    os.makedirs(os.path.join(RECOGNITION_MODELS_PATH, running_name), exist_ok=True)
    if log is None:
        log = create_log(running_name, os.path.join(RECOGNITION_MODELS_PATH, running_name))
    try:
        running_data = train(running_name,
              train_datasets_names,
              val_datasets_names,
              language_name,
              LANGUAGES_PATH,
              DATASETS_VIEWS_PATH,
              DATASETS_PATH,
              RECOGNITION_MODELS_PATH,
              log,
              num_epochs,
              batch_size,
              lr,
              loss_func_name,
              random_transformation,
              max_word_length,
              gpu_ids,
              encoder_type,
              hidden_size,
              num_layers,
              num_heads_self_attention,
              dropout_self_attention,
              without_rotated_data,
              pretrained_recognition_name,
              image_size,
              recognition_resize_method,
              already_normalized,
              decoder_type,
              positional_enconding_type,
              init_pos_as_const,
              enable_icon_or_emoji_text,
              num_workers,
              early_stop_by
              )
        log.info('Recognition training finished!')
        return running_data
    except Exception as e:
        log.exception(str(e))
        raise e


def inference_recognition(inference_folder, training_name, batch_size=16, gpu_ids='0', model=None, model_data=None, num_workers=1, recognition_resize_method=None, uda_data=False):
    if not os.path.isdir(os.path.join(RECOGNITION_MODELS_PATH, training_name)):
        raise ProjectIsNotExists()
    if model is None:
        model = os.path.join(RECOGNITION_MODELS_PATH, training_name, 'recognition.pt')
    if model_data is None:
        model_data = torch.load(os.path.join(RECOGNITION_MODELS_PATH, training_name, 'running_data.pt'), map_location='cpu')
    image_paths_to_text = inference(inference_folder,
                             model,
                             model_data,
                             batch_size,
                             gpu_ids,
                             num_workers,
                             recognition_resize_method=recognition_resize_method)
    return image_paths_to_text




