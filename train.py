import torch
from Experiments.multiexperiments_detection_and_recognition import *
import sys, os
from Config.setting import *


num_gpus = torch.cuda.device_count()
gpu_ids = ','.join([str(i) for i in range(num_gpus)])


training_type = sys.argv[1]
name = sys.argv[2]
crop_size = (int(sys.argv[3]), int(sys.argv[3]))
uda = eval(sys.argv[4])
pretrained_model = sys.argv[5]
uda_train_only_discriminator = int(sys.argv[6])
if pretrained_model == "None":
    pretrained_model = None


if name == 'type_2_type_3':
    dataset_names = [fn for fn in os.listdir(DATASETS_PATH) if 'real' not in fn.lower() and '.zip' not in fn.lower()]
elif name == 'type_1_type_2_type_3':
    dataset_names = [fn for fn in os.listdir(DATASETS_PATH) if '.zip' not in fn.lower()]
elif name == 'type_1':
    dataset_names = [fn for fn in os.listdir(DATASETS_PATH) if 'real' in fn.lower() and '.zip' not in fn.lower()]
external_dataset_uda_names = [os.path.join(fn, 'photos') for fn in os.listdir(EXTERNAL_DATASETS_OCR_PATH) if '.zip' not in fn.lower()]


if training_type == "Detections" and not uda:
    a = MultiExperimentDetectionRecognition(name='detection_' + name)
    dec = a.running_detection_training_configurations(
        [fn for fn in dataset_names if 'train' in fn.lower()],
        [fn for fn in dataset_names if 'val' in fn.lower()],
        detection_model_class="UNet",
        crop=crop_size,
        detection_num_epochs=400,
        detection_batch_size=32 * num_gpus,
        detection_lr=1e-4,
        detection_gpu_ids=gpu_ids,
        num_workers=32,
        random_transformation_probs=(0.25,),
        detection_gamma_scheduler=0.75,
        detection_num_epochs_scheduler=50,
        pretrained_detection_name=pretrained_model
    )
elif training_type == "Detections" and uda:
    a = MultiExperimentDetectionRecognition(name='detection_uda_' + name)
    dec = a.running_detection_training_configurations(
        [fn for fn in dataset_names if 'train' in fn.lower()],
        [fn for fn in dataset_names if 'val' in fn.lower()],
        detection_model_class="UNetUDA",
        crop=crop_size,
        detection_num_epochs=400,
        detection_batch_size=32 * num_gpus,
        detection_lr=1e-4,
        detection_gpu_ids=gpu_ids,
        num_workers=32,
        random_transformation_probs=(0.25,),
        detection_gamma_scheduler=0.75,
        detection_num_epochs_scheduler=50,
        pretrained_detection_name=pretrained_model,
        train_uda=True,
        lambda_uda=10,
        uda_train_only_discriminator=uda_train_only_discriminator,
        accumulate=1,
        src_uda_folder=os.path.join(EXTERNAL_DATASETS_OCR_PATH, external_dataset_uda_names[0])
    )
else:
    a = MultiExperimentDetectionRecognition(name='recognition_' + name)
    rec_nll = a.running_recognition_training_configurations(
        [fn for fn in dataset_names if 'train' in fn.lower()],
        [fn for fn in dataset_names if 'val' in fn.lower()],
        charset=string.ascii_letters + string.digits + string.punctuation,
        direction="ltr",
        charset_to_charset_normalization=None,
        recognition_num_epochs=300,
        loss_functions=('nll',),  # Options: 'nll' or 'ctc'
        encoder_types=('transformer',),
        # Options: 'transformer' which is a transformer encoder or 'bilstm'
        decoder_types=('fc',),
        # Options: 'fc' (regular fully connected header) or 'transformer' which is the transformer decoder.
        # I found the transformer decoder to heavy for this task, so I run only with 'fc'
        positional_enconding_types=("const",),
        # Options: "const" for constant positional encoding or "embedding" for learnable positional encoding
        init_options=(True,),
        # Init positional encoding to sinusodial encoding. Should be True for constant positional encoding. In the case
        # of learnable positional encoding, True will initialize the positional encoding to the sinusodial encoding while False
        # will initialize it to a random encoding.
        random_transformation_probs=(0.25,),
        # On-Training augmentations probability
        num_workers=16,
        recognition_batch_size=512 * num_gpus,
        recognition_resize_method="ar",  # Options:
        recognition_lr=1e-4,
        recognition_gpu_ids=gpu_ids,
        recognition_hidden_size=512,
        recognition_num_layers=4,
        recognition_num_heads_self_attention=16,
        recognition_dropout_self_attention=0.0,
        recognition_max_word_length=38,
        recognition_image_size=(48, 150),
        pretrained_recognition_name=pretrained_model,
        # Initialize the model to a former learned recognition model
        enable_icon_or_emoji_text=False,
        # Do not train on samples with emojis or icons
    )
