import gc
import shutil
import torch
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data.dataloader import Dataset, DataLoader
import os
from torchvision.io import read_image
from Data.dataset_view import OCRDatasetView
from ModelUtils.model_utils import *
from tqdm import tqdm
import numpy as np
import time
import uuid
from Config.setting import *
from torch.nn.utils.rnn import pad_sequence
from GeneralUtils.utils import *
from Recognition.model import RecognitionModel


def resize_and_norm_image(image, recognition_resize_method, image_size, resize_transformation_without_ar, norm, already_normalized):
    need_resize_or_padding = (image.shape[1] != image_size[0]) or (image.shape[2] != image_size[1])
    if need_resize_or_padding and recognition_resize_method == 'without_ar':
        image = resize_transformation_without_ar(image)
    elif need_resize_or_padding and recognition_resize_method == 'ar':
        if image.shape[1] > image_size[0]:
            new_width = max(int(float(image_size[0]) / float(image.shape[1]) * image.shape[2]), 1)
            image = torchvision.transforms.Resize((image_size[0], new_width))(image)
        if image.shape[2] > image_size[1]:
            new_height = max(int(float(image_size[1]) / float(image.shape[2]) * image.shape[1]), 1)
            image = torchvision.transforms.Resize((new_height, image_size[1]))(image)
        if image.shape[1] < image_size[0] and image.shape[2] < image_size[1]:
            if (float(image.shape[1]) / float(image_size[0])) > (float(image.shape[2]) / float(image_size[1])):
                resize_transform = torchvision.transforms.Resize((image_size[0], min(max(int((float(image_size[0]) / float(image.shape[1])) * image.shape[2]), 1), image_size[1])))
            else:
                resize_transform = torchvision.transforms.Resize((min(max(int((float(image_size[1]) / float(image.shape[2])) * image.shape[1]), 1), image_size[0]), image_size[1]))
            image = resize_transform(image)
    elif need_resize_or_padding and recognition_resize_method == 'comb':
        if image.shape[1] < image_size[0] and image.shape[2] < image_size[1]:
            if (float(image.shape[1]) / float(image_size[0])) > (float(image.shape[2]) / float(image_size[1])):
                resize_transform = torchvision.transforms.Resize((image_size[0], min(max(int((float(image_size[0]) / float(image.shape[1])) * image.shape[2]), 1), image_size[1])))
            else:
                resize_transform = torchvision.transforms.Resize((min(max(int((float(image_size[1]) / float(image.shape[2])) * image.shape[1]), 1), image_size[0]), image_size[1]))
            image = resize_transform(image)
        if image_size[0] < image.shape[1] or image_size[1] < image.shape[2]:
            image = resize_transformation_without_ar(image)
    elif need_resize_or_padding and recognition_resize_method == 'dynamic_w':
        resize_transform = torchvision.transforms.Resize((image_size[0], max(int((float(image_size[0]) / float(image.shape[1])) * image.shape[2]), 1)))
        image = resize_transform(image)
    elif need_resize_or_padding and recognition_resize_method == 'ar_clip_w':
        if image.shape[1] > image_size[0]:
            new_width = max(int(float(image_size[0]) / float(image.shape[1]) * image.shape[2]), 1)
            image = torchvision.transforms.Resize((image_size[0], new_width))(image)
        image = image[:, :, :min(image.shape[2], image_size[1])]
    elif need_resize_or_padding and recognition_resize_method == 'rosetta_training':
        # This is the resize method from the Rosetta paper, But without the stretching they proposed (stretching images with constant factor of 1.2)
        if image.shape[1] > image_size[0] or image.shape[2] > image_size[1]:
            image = resize_transformation_without_ar(image)
    elif need_resize_or_padding and recognition_resize_method == 'rosetta_inference':
        resize_transform = torchvision.transforms.Resize((image_size[0], max(int((float(image_size[0]) / float(image.shape[1])) * image.shape[2]), 1)))
        image = resize_transform(image)
    image = image / 255.0
    if not already_normalized:
        image = norm(image)
    if need_resize_or_padding and (image_size[0] > image.shape[1] or image_size[1] > image.shape[2]) and recognition_resize_method != 'dynamic_w':
        image = F.pad(image, (0, image_size[1] - image.shape[2], 0, image_size[0] - image.shape[1]))
    return image


class VariableWidthCollate:
    METHODS = ["dynamic_w", "rosetta_inference"]

    def __init__(self, image_size, padding_value):
        self.image_size = image_size
        self.padding_value = padding_value

    def pad_collate_dynamic_w(self, batch):
        if len(batch[0]) == 4:
            (x, y, lens, paths) = zip(*batch)
            max_w = max([x[i].shape[2] for i in range(len(x))])
            x = [F.pad(img, (0, max_w - img.shape[2], 0, self.image_size[0] - img.shape[1])) for img in x]
            x = torch.stack(x, dim=0)
            y = pad_sequence(y, batch_first=True, padding_value=self.padding_value)
            lens = torch.Tensor(lens)
            return x, y, lens, paths
        else:
            (x, paths) = zip(*batch)
            max_w = max([x[i].shape[2] for i in range(len(x))])
            x = [F.pad(img, (0, max_w - img.shape[2], 0, self.image_size[0] - img.shape[1])) for img in x]
            x = torch.stack(x, dim=0)
            return x, paths


class RecognitionDataset(Dataset):
    def __init__(self, image_paths_to_recognition_data, random_transformation, charset,
                 without_rotated_data=False, max_word_length=38, inference=False,
                 image_size=(48, 150), recognition_resize_method='ar', loss_func_name='nll', already_normalized=False):
        self.image_paths_to_recognition_data = image_paths_to_recognition_data
        if not without_rotated_data or inference:
            self.image_paths = list(self.image_paths_to_recognition_data.keys())
        else:
            self.image_paths = list([p for p in self.image_paths_to_recognition_data.keys() if ('rotated' not in self.image_paths_to_recognition_data[p].keys() or not self.image_paths_to_recognition_data[p]['rotated'])])
        self.random_transformation = random_transformation
        self.charset = list(charset) + ['']
        self.max_word_length = max_word_length
        self.inference = inference
        self.image_size = image_size
        self.recognition_resize_method = recognition_resize_method
        self.norm = torchvision.transforms.Normalize(0, 1)
        self.loss_func_name = loss_func_name
        self.already_normalized = already_normalized
        self.transformations = [
            torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3),
            torchvision.transforms.GaussianBlur((3, 3), (0.1, 2)),
            torchvision.transforms.RandomPosterize(bits=1, p=1.0),
            torchvision.transforms.RandomSolarize(120, p=1.0),
            torchvision.transforms.RandomInvert(p=1.0),
            torchvision.transforms.Resize(self.image_size)
        ]
        self.resize_transformation_without_ar = torchvision.transforms.Resize(self.image_size)

    def indexes_to_string(self, indexes):
        if self.loss_func_name == 'nll':
            text = ''.join([self.charset[idx] for idx in indexes])
        elif self.loss_func_name == 'ctc':
            text = ''.join([self.charset[indexes[i]] for i in range(len(indexes)) if ((i == 0) or (indexes[i] != indexes[i - 1]))])
        return text

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = read_image(image_path, torchvision.io.ImageReadMode.RGB)
        if not self.inference:
            text = self.image_paths_to_recognition_data[image_path]['text']
            text = [self.charset.index(ch) for ch in text]
            text_length = len(text)
            if self.recognition_resize_method != 'dynamic_w':
                text += [self.charset.index('')] * (self.max_word_length - len(text))
            text = torch.LongTensor(text)
            if self.random_transformation > 0.0 and image.shape[1] > 3 and image.shape[2] > 3:
                for transform in self.transformations:
                    if np.random.rand() < self.random_transformation:
                        image = transform(image)
        image = resize_and_norm_image(image, self.recognition_resize_method, self.image_size, self.resize_transformation_without_ar, None, True)
        if not self.inference:
            return image, text, text_length, image_path
        return image, image_path

    def __len__(self):
        return len(self.image_paths)


def train(running_name, train_datasets_names, val_datasets_names, language_name, language_folder, datasets_views_folder, datasets_folder, models_folder,
          log, num_epochs=100, batch_size=256, lr=7e-5, loss_func_name='nll', random_transformation=0.25, max_word_length=1000, gpu_ids='0',
          encoder_type='bilstm', hidden_size=256, num_layers=4, num_heads_self_attention=4,
          dropout_self_attention=0.0, without_rotated_data=True, pretrained_recognition_name=None,
          image_size=(32, 100), recognition_resize_method='ar', already_normalized=False, decoder_type='fc', positional_enconding_type='const', init_pos_as_const=True, enable_icon_or_emoji_text=False,
          num_workers=1, early_stop_by=('val_loss', 'min')):
    start_time = time.time()
    device = get_device(gpu_ids)
    os.makedirs(models_folder, exist_ok=True)
    running_folder = os.path.join(models_folder, running_name)
    os.makedirs(running_folder, exist_ok=True)
    if not os.path.isfile(os.path.join(datasets_views_folder, running_name + '_recognition_train.json')):
        train_data_view = OCRDatasetView(running_name + '_recognition_train', datasets_views_folder, datasets_folder, language_name=language_name, language_folder=language_folder, running_names=train_datasets_names,
                                     detection=False, recognition=True, layout=False, max_word_length=max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
    else:
        log.info('Load existing train data view')
        train_data_view = OCRDatasetView(running_name + '_recognition_train', datasets_views_folder, datasets_folder, load_view_file=True)
        log.info('Finished loading')
    image_paths_to_recognition_data = train_data_view.image_paths_to_recognition_data
    charset = train_data_view.language.charset
    if not os.path.isfile(os.path.join(datasets_views_folder, running_name + '_recognition_val.json')):
        val_data_view = OCRDatasetView(running_name + '_recognition_val', datasets_views_folder, datasets_folder, language_name=language_name, language_folder=language_folder, running_names=val_datasets_names,
                                   detection=False, recognition=True, layout=False, max_word_length=max_word_length, enable_icon_or_emoji_text=enable_icon_or_emoji_text)
    else:
        log.info('Load existing val data view')
        val_data_view = OCRDatasetView(running_name + '_recognition_val', datasets_views_folder, datasets_folder, load_view_file=True)
        log.info('Finished loading')
    val_image_paths_to_recognition_data = val_data_view.image_paths_to_recognition_data
    if pretrained_recognition_name is None:
        recognition_model = RecognitionModel(len(charset),
                                             encoder_type=encoder_type, decoder_type=decoder_type, hidden_size=hidden_size, num_layers=num_layers,
                                             num_heads_self_attention=num_heads_self_attention,
                                             dropout_self_attention=dropout_self_attention,
                                             in_channels=3, max_word_length=max_word_length, positional_enconding_type=positional_enconding_type,
                                             init_pos_as_const=init_pos_as_const)

        recognition_model = load_model_by_gpu_ids(recognition_model, device=device)
    else:
        recognition_model = load_model_by_gpu_ids(os.path.join(models_folder, pretrained_recognition_name, 'recognition.pt'), device=device)
    recognition_model.train()
    recognition_model_num_parameters = get_number_of_parameters(recognition_model)
    optimizer = torch.optim.Adam(recognition_model.parameters(), lr=lr)
    variable_width_handler = VariableWidthCollate(image_size, len(charset))
    collate_fn = None if recognition_resize_method not in variable_width_handler.METHODS else variable_width_handler.pad_collate_dynamic_w
    dataset = RecognitionDataset(image_paths_to_recognition_data, random_transformation, charset, without_rotated_data=without_rotated_data,
                                 max_word_length=max_word_length, image_size=image_size,
                                 recognition_resize_method=recognition_resize_method,
                                 loss_func_name=loss_func_name, already_normalized=already_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=True)
    log.info('Train Dataset Size : {}'.format(len(dataset)))
    val_dataset = RecognitionDataset(val_image_paths_to_recognition_data, random_transformation=random_transformation, charset=charset, without_rotated_data=without_rotated_data,
                                     max_word_length=max_word_length, image_size=image_size,
                                     recognition_resize_method=recognition_resize_method,
                                     loss_func_name=loss_func_name, already_normalized=already_normalized)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=True)
    log.info('Val Dataset Size : {}'.format(len(val_dataset)))
    if loss_func_name == 'nll':
        loss_func = nn.CrossEntropyLoss(reduction='sum')
    elif loss_func_name == 'ctc':
        loss_func = nn.CTCLoss(blank=len(charset), zero_infinity=True, reduction='sum')
    if os.path.isfile(os.path.join(running_folder, 'running_data.pt')):
        log.info('Recognition training resume!')
        checkpoint = torch.load(os.path.join(running_folder, 'running_data.pt'), map_location='cpu')
        recognition_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['current_epoch'] + 1
        start_time = start_time - checkpoint['duration']
        best_val_epoch = checkpoint['best_val_epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        val_evaluation = checkpoint['val_evaluation']
        # val_evaluation_before_training = checkpoint['val_evaluation_before_training']
        log.info(f'Continue to train on {num_epochs - starting_epoch} more epochs')
        if num_epochs == 0:
            return checkpoint
    else:
        best_val_epoch = -1
        starting_epoch = 0
        train_loss = []
        val_loss = []
        val_evaluation = []
        # val_ep_loss, val_ep_preds_paths_to_str, val_ep_targets_paths_to_str = val(recognition_model, loss_func, loss_func_name, val_dataset, val_dataloader, device)
        # val_evaluation_before_training = evaluate_recognition(val_ep_preds_paths_to_str, val_ep_targets_paths_to_str)
        # val_evaluation_before_training['val_loss'] = val_ep_loss
    recognition_model = load_model_by_gpu_ids(recognition_model, gpu_ids)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters=num_epochs, last_epoch=starting_epoch - 1, verbose=True)
    onnx_ok = None
    # scaler = torch.cuda.amp.GradScaler()
    # device_str = 'cpu' if gpu_ids == '-1' else 'cuda'
    for epoch in range(starting_epoch, num_epochs):
        gc.collect()
        loss_ep = 0.0
        recognition_model.train()
        for images, targets, lengths, paths in tqdm(dataloader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            # with torch.autocast(device_type=device_str, dtype=torch.float16):
            preds = recognition_model(images, targets)
            if loss_func_name == 'nll':
                loss = loss_func(preds, targets[:, :preds.shape[2]])
            elif loss_func_name == 'ctc':
                lengths = lengths.to(device)
                loss = loss_func(preds.permute((2, 0, 1)), targets[:, :preds.shape[2]], torch.LongTensor([preds.shape[2]] * preds.shape[0]), lengths)
            loss.backward()
            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss_ep += loss.item()
        train_loss.append(loss_ep / float(len(dataset)))
        val_ep_loss = val(recognition_model, loss_func, loss_func_name, val_dataset, val_dataloader, device)  #, val_ep_preds_paths_to_str, val_ep_targets_paths_to_str
        ep_val_evaluation = {}  #evaluate_recognition(val_ep_preds_paths_to_str, val_ep_targets_paths_to_str)
        ep_val_evaluation['val_loss'] = val_ep_loss
        val_evaluation.append(ep_val_evaluation)
        val_loss.append(val_ep_loss)
        log.info('Validation scores : {}'.format(ep_val_evaluation))
        if (early_stop_by[1] == 'min' and min([x[early_stop_by[0]] for x in val_evaluation]) == val_evaluation[-1][early_stop_by[0]]) or (early_stop_by[1] == 'max' and max([x[early_stop_by[0]] for x in val_evaluation]) == val_evaluation[-1][early_stop_by[0]]):
            log.info('Saving as best epoch so far by {} the {}'.format(early_stop_by[1], early_stop_by[0]))
            save_pytorch_model(recognition_model, os.path.join(running_folder, 'recognition.pt'))
            # save_onnx_model(torch.randn(images.shape, requires_grad=True, dtype=images.dtype).to(device), recognition_model,
            #                 os.path.join(running_folder, 'recognition.onnx'), dynamic_width=True)
            # onnx_ok = compare_onnx_and_pytorch_models(load_model_by_gpu_ids(os.path.join(running_folder, 'recognition.onnx')), recognition_model, device, images, random=True, decimal_point=4, log=log)
            best_val_epoch = epoch
        duration = time.time() - start_time
        running_data = {
            'running_name': running_name,
            'train_datasets_names': train_datasets_names,
            'val_datasets_names': val_datasets_names,
            'charset': charset,
            'max_word_length': max_word_length,
            'datasets_views_folder': datasets_views_folder,
            'datasets_folder': datasets_folder,
            'models_folder': models_folder,
            'num_epochs': num_epochs,
            'current_epoch': epoch,
            'best_val_epoch': best_val_epoch,
            'batch_size': batch_size,
            'loss_func_name': loss_func_name,
            'lr': lr,
            'duration': duration,
            'num_parameters': recognition_model_num_parameters,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_evaluation': val_evaluation,
            # 'val_evaluation_before_training': val_evaluation_before_training,
            'early_stop_by': early_stop_by,
            'num_samples_train': len(dataset),
            'num_samples_val': len(val_dataset),
            'gpu_ids': gpu_ids,
            'encoder_type': encoder_type,
            'decoder_type': decoder_type,
            'positional_enconding_type': positional_enconding_type,
            'init_pos_as_const': init_pos_as_const,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads_self_attention': num_heads_self_attention,
            'dropout_self_attention': dropout_self_attention,
            'without_rotated_data': without_rotated_data,
            'pretrained_recognition_name': pretrained_recognition_name,
            'image_size': image_size,
            'recognition_resize_method': recognition_resize_method,
            'already_normalized': already_normalized,
            'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
            'num_workers': num_workers,
            'random_transformation': random_transformation,
            'model_state_dict': get_model_state_dict(recognition_model),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'onnx_ok': onnx_ok
        }
        torch.save(running_data, os.path.join(running_folder, 'running_data.pt'))
        log.info('Epoch : {}, Train : {}, Val : {}'.format(epoch, train_loss[-1], val_loss[-1]))
        lr_scheduler.step()
    return running_data


@torch.no_grad()
def val(recognition_model, loss_func, loss_func_name, dataset, dataloader, device):
    recognition_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets, lengths, paths in tqdm(dataloader):
            images, targets = images.to(device), targets.to(device)
            preds = recognition_model(images, targets)
            if loss_func_name == 'nll':
                loss = loss_func(preds, targets[:, :preds.shape[2]]).detach()
            elif loss_func_name == 'ctc':
                loss = loss_func(preds.permute((2, 0, 1)), targets[:, :preds.shape[2]], torch.LongTensor([preds.shape[2]] * preds.shape[0]), lengths).detach()
            test_loss += loss.item()
            #preds_paths_to_str, targets_paths_to_str = compute_strings(preds, targets, paths, dataset.indexes_to_string)
    test_loss = test_loss / float(len(dataset))
    return test_loss  #, preds_paths_to_str, targets_paths_to_str


@torch.no_grad()
def compute_strings(preds, targets, paths, indexes_to_string_func):
    preds_paths_to_str = {}
    targets_paths_to_str = {}
    _, out = torch.max(preds, dim=1)
    for i in range(out.shape[0]):
        preds_paths_to_str[paths[i]] = indexes_to_string_func(out[i, :])
        targets_paths_to_str[paths[i]] = indexes_to_string_func(targets[i, :])
    return preds_paths_to_str, targets_paths_to_str


@torch.no_grad()
def inference(inference_folder, model, model_data, batch_size, gpu_ids, num_workers, recognition_resize_method=None):
    device = get_device(gpu_ids)
    recognition_model = load_model_by_gpu_ids(model, gpu_ids)
    working_folder = make_tmp_folder()
    if recognition_resize_method is None:
        recognition_resize_method = model_data['recognition_resize_method']
    variable_width_handler = VariableWidthCollate(model_data['image_size'], len(model_data['charset']))
    collate_fn = None if recognition_resize_method not in variable_width_handler.METHODS else variable_width_handler.pad_collate_dynamic_w
    image_paths_to_recognition_data = OCRDatasetView('inference', working_folder, inference_folder, inference=True, detection=False, recognition=True, layout=False).image_paths_to_recognition_data
    dataset = RecognitionDataset(image_paths_to_recognition_data, random_transformation=0.0, charset=model_data['charset'], inference=True,
                                 image_size=model_data['image_size'], recognition_resize_method=recognition_resize_method,
                                 loss_func_name=model_data['loss_func_name'], already_normalized=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    image_paths_to_text = {}
    for images, paths in tqdm(dataloader):
        images = images.to(device)
        preds = inference_model(recognition_model, images)
        _, out = torch.max(preds, dim=1)
        for i in range(out.shape[0]):
            image_paths_to_text[paths[i]] = dataset.indexes_to_string(out[i, :])
    shutil.rmtree(working_folder)
    return image_paths_to_text

