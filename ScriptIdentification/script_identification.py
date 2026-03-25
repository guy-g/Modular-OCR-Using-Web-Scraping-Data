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
from Recognition.recognition import resize_and_norm_image
from ScriptIdentification.model import ScriptClassifier
from GeneralUtils.utils import *


class ScriptDataset(Dataset):
    def __init__(self,
                 image_paths_to_scripts_data,
                 random_transformation,
                 inference=False, image_size=(32, 100),
                 script_resize_method='ar', already_normalized=False):
        self.image_paths_to_scripts_data = image_paths_to_scripts_data
        self.image_paths = list(self.image_paths_to_scripts_data.keys())
        self.random_transformation = random_transformation
        self.inference = inference
        self.image_size = image_size
        self.script_resize_method = script_resize_method
        self.norm = torchvision.transforms.Normalize(0, 1)
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

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = read_image(image_path, torchvision.io.ImageReadMode.RGB)
        if not self.inference:
            multi_class_binary_labels = self.image_paths_to_scripts_data[image_path]
            if self.random_transformation > 0.0 and image.shape[1] > 3 and image.shape[2] > 3:
                for transform in self.transformations:
                    if np.random.rand() < self.random_transformation:
                        image = transform(image)
        image = resize_and_norm_image(image, self.script_resize_method, self.image_size,
                                      self.resize_transformation_without_ar, None, True)
        if not self.inference:
            return image, multi_class_binary_labels, image_path
        return image, image_path

    def __len__(self):
        return len(self.image_paths)


def train(running_name, train_datasets_names, val_datasets_names, language_names, language_folder,
          datasets_views_folder, datasets_folder, models_folder,
          log, num_epochs=100, batch_size=256, lr=7e-5, random_transformation=0.25, gpu_ids='0',
          pretrained_script_name=None, image_size=(48, 150), resize_method='ar', already_normalized=False,
          enable_icon_or_emoji_text=False, num_workers=1):
    start_time = time.time()
    device = get_device(gpu_ids)
    os.makedirs(models_folder, exist_ok=True)
    running_folder = os.path.join(models_folder, running_name)
    os.makedirs(running_folder, exist_ok=True)
    image_paths_to_scripts = {}
    scripts_to_num_samples = {}
    for (datasets_type, datasets_names) in (('train', train_datasets_names), ('val', val_datasets_names)):
        image_paths_to_scripts[datasets_type] = {}
        scripts_to_num_samples[datasets_type] = {}
        data_views = []
        for language_name in language_names:
            if not os.path.isfile(os.path.join(datasets_views_folder,
                                               running_name + '_script_{}_{}.json'.format(datasets_type,
                                                                                          language_name))):
                data_view = OCRDatasetView(running_name + '_script_{}_{}'.format(datasets_type, language_name),
                                           datasets_views_folder, datasets_folder,
                                           language_name=language_name, language_folder=language_folder,
                                           running_names=datasets_names,
                                           detection=False, recognition=True, layout=False,
                                           enable_icon_or_emoji_text=enable_icon_or_emoji_text, ignore_direction=True)
            else:
                log.info('Load existing data view')
                data_view = OCRDatasetView(running_name + '_script_{}_{}'.format(datasets_type, language_name),
                                           datasets_views_folder,
                                           datasets_folder, load_view_file=True)
                log.info('Finished loading')
            log.info('{} - {} : {} samples'.format(datasets_type, language_name, len(data_view.image_paths_to_recognition_data.keys())))
            scripts_to_num_samples[datasets_type][language_name] = len(data_view.image_paths_to_recognition_data.keys())
            data_views.append(data_view)
        for data_view in data_views:
            for k, v in data_view.image_paths_to_recognition_data.items():
                if k not in image_paths_to_scripts[datasets_type].keys():
                    image_paths_to_scripts[datasets_type][k] = {dv.language_name: 0 for dv in data_views}
                image_paths_to_scripts[datasets_type][k][data_view.language_name] = 1

    if pretrained_script_name is None:
        script_model = ScriptClassifier(scripts=language_names, in_channels=3)
        script_model = load_model_by_gpu_ids(script_model, device=device)
    else:
        script_model = load_model_by_gpu_ids(os.path.join(models_folder, pretrained_script_name, 'script_identification.pt'), device=device)
    script_model.train()
    script_model_num_parameters = get_number_of_parameters(script_model)
    optimizer = torch.optim.Adam(script_model.parameters(), lr=lr)
    dataset = ScriptDataset(image_paths_to_scripts['train'],
                            random_transformation=random_transformation,
                            image_size=image_size,
                            script_resize_method=resize_method,
                            already_normalized=already_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    log.info('Train Dataset Size : {}'.format(len(dataset)))
    val_dataset = ScriptDataset(image_paths_to_scripts['val'],
                            random_transformation=random_transformation,
                            image_size=image_size,
                            script_resize_method=resize_method,
                            already_normalized=already_normalized)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    log.info('Val Dataset Size : {}'.format(len(val_dataset)))
    loss_func = nn.BCELoss()
    if os.path.isfile(os.path.join(running_folder, 'running_data.pt')):
        log.info('Script Identification training resume!')
        checkpoint = torch.load(os.path.join(running_folder, 'running_data.pt'), map_location='cpu')
        script_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['current_epoch'] + 1
        start_time = start_time - checkpoint['duration']
        best_val_epoch = checkpoint['best_val_epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        log.info(f'Continue to train on {num_epochs - starting_epoch} more epochs')
        if num_epochs == 0:
            return checkpoint
    else:
        best_val_epoch = -1
        starting_epoch = 0
        train_loss = []
        val_loss = []
    script_model = load_model_by_gpu_ids(script_model, gpu_ids)
    onnx_ok = None
    for epoch in range(starting_epoch, num_epochs):
        gc.collect()
        loss_ep = 0.0
        script_model.train()
        for images, targets, paths in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            preds = script_model(images)
            loss = 0.0
            for k in targets.keys():
                loss += loss_func(preds[k], targets[k].float().to(device))
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
        train_loss.append(loss_ep / float(len(dataset)))
        val_loss.append(val(script_model, loss_func, val_dataloader, device) / float(len(val_dataset)))
        if min(val_loss) == val_loss[-1]:
            save_pytorch_model(script_model, os.path.join(running_folder, 'script_identification.pt'))
            # save_onnx_model(torch.randn(images.shape, requires_grad=True, dtype=images.dtype).to(device), script_model,
            #                 os.path.join(running_folder, 'script_identification.onnx'), dynamic_width=True)
            # onnx_ok = compare_onnx_and_pytorch_models(load_model_by_gpu_ids(os.path.join(running_folder, 'script_identification.onnx')), script_model, device, images, random=True, decimal_point=4, log=log)
            best_val_epoch = epoch
        duration = time.time() - start_time
        running_data = {
            'running_name': running_name,
            'train_datasets_names': train_datasets_names,
            'val_datasets_names': val_datasets_names,
            'datasets_views_folder': datasets_views_folder,
            'datasets_folder': datasets_folder,
            'models_folder': models_folder,
            'language_names': language_names,
            'num_epochs': num_epochs,
            'current_epoch': epoch,
            'best_val_epoch': best_val_epoch,
            'batch_size': batch_size,
            'lr': lr,
            'duration': duration,
            'num_parameters': script_model_num_parameters,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'num_samples_train': len(dataset),
            'num_samples_val': len(val_dataset),
            'scripts_to_num_samples': scripts_to_num_samples,
            'gpu_ids': gpu_ids,
            'pretrained_script_name': pretrained_script_name,
            'image_size': image_size,
            'resize_method': resize_method,
            'already_normalized': already_normalized,
            'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
            'num_workers': num_workers,
            'random_transformation': random_transformation,
            'model_state_dict': get_model_state_dict(script_model),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'onnx_ok': onnx_ok
        }
        torch.save(running_data, os.path.join(running_folder, 'running_data.pt'))
        log.info('Epoch : {}, Train : {}, Val : {}'.format(epoch, train_loss[-1], val_loss[-1]))
    return running_data


def val(script_model, loss_func, dataloader, device):
    script_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = images.to(device)
            preds = script_model(images)
            loss = 0.0
            for k in targets.keys():
                loss += loss_func(preds[k], targets[k].float().to(device))
            test_loss += loss.item()
    return test_loss


@torch.no_grad()
def inference(inference_folder, model, model_data, binary_threshold, batch_size, gpu_ids, num_workers):
    device = get_device(gpu_ids)
    script_model = load_model_by_gpu_ids(model, gpu_ids)
    working_folder = make_tmp_folder()
    image_paths_to_recognition_data = OCRDatasetView('inference', working_folder, inference_folder, inference=True,
                                                     detection=False, recognition=True,
                                                     layout=False).image_paths_to_recognition_data
    dataset = ScriptDataset(image_paths_to_recognition_data,
                            random_transformation=0.0,
                            inference=True,
                            image_size=model_data['image_size'],
                            script_resize_method=model_data['resize_method'],
                            already_normalized=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    image_paths_to_scripts = {}
    for images, paths in tqdm(dataloader):
        images = images.to(device)
        preds = inference_model(script_model, images, model_data['language_names'])
        for k in preds.keys():
            preds[k] = (preds[k] > binary_threshold) * 1.0
        for i in range(len(paths)):
            image_paths_to_scripts[paths[i]] = {k: preds[k][i].item() for k in preds.keys()}
    shutil.rmtree(working_folder)
    return image_paths_to_scripts

