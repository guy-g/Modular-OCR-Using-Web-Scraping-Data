import time

import torch.optim.lr_scheduler
from torch.utils.data.dataloader import Dataset, DataLoader
from Data.dataset_view import OCRDatasetView, LightOCRDatasetView
from ModelUtils.model_utils import *
from tqdm import tqdm
from GeneralUtils.utils import *
from Detection.model import *
from Detection.detection_dataset import DetectionDataset
from PIL import Image


def train(running_name, train_datasets_names, val_datasets_names, datasets_views_folder,
          datasets_folder, models_folder, log,
          detection_data_classes_field_names, detection_data_fields_to_mask_in_loss=None,
          rectangular_bounding_boxes=True, weight_by_class_num_pixels=True,
          classes_colors=((0, 0, 0), (1, 1, 1)), num_epochs=100, batch_size=16, lr=1e-5,
          crop=(320, 320), depth=6, gpu_ids='0', binary_mask_threshold=0.5, pretrained_detection_name=None, consider_perspective=True, consider_elastic=True, detection_data_ready=False,
          mask_in_loss_first=True, enable_icon_or_emoji_text=False, num_workers=1, random_transformation=0.25,
          gamma_scheduler=0.1, num_epochs_scheduler=120, model_class="UNet", accumulate=1, train_uda=True, src_uda_folder=r'',
          lambda_uda=1, uda_train_only_discriminator=0):
    start_time = time.time()
    device = get_device(gpu_ids)
    os.makedirs(models_folder, exist_ok=True)
    running_folder = os.path.join(models_folder, running_name)
    os.makedirs(running_folder, exist_ok=True)
    validation_example_folder = os.path.join(running_folder, 'validation_example')
    os.makedirs(validation_example_folder, exist_ok=True)
    num_classes = len(detection_data_classes_field_names) + 1  # + 1 for background
    classes_colors = [torch.LongTensor([channel_color * 255 for channel_color in c]).view((3, 1)) for c in classes_colors]
    if not enable_icon_or_emoji_text:
        if not os.path.isfile(os.path.join(datasets_views_folder, running_name + '_detection_train.json')):
            train_dataset_view = OCRDatasetView(running_name + '_detection_train', datasets_views_folder, datasets_folder,
                                                running_names=train_datasets_names, enable_icon_or_emoji_text=enable_icon_or_emoji_text, recognition=False, layout=False)
        else:
            log.info('Load existing train data view')
            train_dataset_view = OCRDatasetView(running_name + '_detection_train', datasets_views_folder, datasets_folder, load_view_file=True)
            log.info('Finished loading')
        if not os.path.isfile(os.path.join(datasets_views_folder, running_name + '_detection_val.json')):
            val_dataset_view = OCRDatasetView(running_name + '_detection_val', datasets_views_folder, datasets_folder,
                                          running_names=val_datasets_names, enable_icon_or_emoji_text=enable_icon_or_emoji_text, recognition=False, layout=False)
        else:
            log.info('Load existing val data view')
            val_dataset_view = OCRDatasetView(running_name + '_detection_val', datasets_views_folder, datasets_folder, load_view_file=True)
            log.info('Finished loading')
    else:
        train_dataset_view = LightOCRDatasetView(
                running_name + '_detection_train',
                running_names=train_datasets_names,
                detection=True,
                recognition=False
            )
        val_dataset_view = LightOCRDatasetView(
                running_name + '_detection_val',
                running_names=val_datasets_names,
                detection=True,
                recognition=False
            )

    if train_uda:
        batch_size = batch_size // 2

    dataset = DetectionDataset(train_dataset_view.image_paths_to_detection_data, detection_data_classes_field_names, detection_data_fields_to_mask_in_loss,
                               rectangular_bounding_boxes, crop, depth, False, consider_perspective=consider_perspective, consider_elastic=consider_elastic,
                               detection_data_ready=detection_data_ready, mask_in_loss_first=mask_in_loss_first, random_transformation=random_transformation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, drop_last=True)
    log.info('Train Dataset Size : {}'.format(len(dataset)))
    val_dataset = DetectionDataset(val_dataset_view.image_paths_to_detection_data, detection_data_classes_field_names, detection_data_fields_to_mask_in_loss,
                                   rectangular_bounding_boxes, crop, depth, False, consider_perspective=consider_perspective, consider_elastic=consider_elastic,
                                   detection_data_ready=detection_data_ready, mask_in_loss_first=mask_in_loss_first, random_transformation=random_transformation)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, drop_last=True)

    if train_uda:
        inference_dataset_view = OCRDatasetView('train_unlabeled_dataset', src_uda_folder, src_uda_folder, inference=True, detection=True, recognition=False, layout=False, to_save=False)
        train_unlabeled_dataset = DetectionDataset(inference_dataset_view.image_paths_to_detection_data, detection_data_classes_field_names, crop=crop, depth=depth, inference=True, random_transformation=random_transformation,
                                                   duplicate_data_to=len(dataset))
        train_unlabeled_dataloader = DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    log.info('Val Dataset Size : {}'.format(len(val_dataset)))
    if pretrained_detection_name is None:
        if model_class == "UNet":
            segmentation_model = UNet(n_blocks=depth, out_channels=num_classes)
        elif model_class == "UNetUDA":
            segmentation_model = UnetUDA({'n_blocks': depth, 'out_channels': num_classes})
        else:
            segmentation_model = Unet2D_Diffusers(n_blocks=depth, out_channels=num_classes)
        segmentation_model = load_model_by_gpu_ids(segmentation_model, device=device)
    else:
        pretrained_detection = load_model_by_gpu_ids(os.path.join(models_folder, pretrained_detection_name, 'segmentation.pt'), device=device)
        if model_class == "UNetUDA":
            segmentation_model = UnetUDA({'n_blocks': depth, 'out_channels': num_classes})
            if pretrained_detection.__class__.__name__ == 'UnetUDA':
                segmentation_model = pretrained_detection
            else:
                segmentation_model.unet = pretrained_detection
            segmentation_model = load_model_by_gpu_ids(segmentation_model, device=device)
        else:
            if pretrained_detection.__class__.__name__ == 'UnetUDA':
                segmentation_model = pretrained_detection.unet
            else:
                segmentation_model = pretrained_detection
    # if train_uda:
        # discriminator = Discriminator(in_channels=2, start_filters=64)
        # discriminator = load_model_by_gpu_ids(discriminator, device=device)
    segmentation_model.train()
    # discriminator.train()
    segmentation_model_num_parameters = get_number_of_parameters(segmentation_model)
    if weight_by_class_num_pixels:
        relative_class_freq = [float(train_dataset_view.num_pixels_each_class['detection'][k]) / float(sum(train_dataset_view.num_pixels_each_class['detection'])) for k in range(num_classes)]
        inverse_relative_class_freq = [(1.0 - k) for k in relative_class_freq]
        loss_weight = torch.Tensor(inverse_relative_class_freq)
    else:
        loss_weight = torch.Tensor([(1.0 / float(num_classes))] * num_classes)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight.to(device))
    if os.path.isfile(os.path.join(running_folder, 'running_data.pt')):
        log.info('Detection training resume!')
        checkpoint = torch.load(os.path.join(running_folder, 'running_data.pt'), map_location='cpu')
        segmentation_model.load_state_dict(checkpoint['model_state_dict'])
        starting_epoch = checkpoint['current_epoch'] + 1
        start_time = start_time - checkpoint['duration']
        best_val_epoch = checkpoint['best_val_epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        adv_loss_per_epoch = checkpoint['adv_loss_per_epoch']
        print(uda_train_only_discriminator, starting_epoch)
        #time.sleep(20)
        if uda_train_only_discriminator < starting_epoch:
            optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=lr)
        elif train_uda:
            optimizer = torch.optim.Adam(segmentation_model.discriminator.parameters(), lr=lr)
            for p in segmentation_model.unet.parameters():
                p.requires_grad = False
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log.info(f'Continue to train on {num_epochs - starting_epoch} more epochs')
        if num_epochs == 0:
            return checkpoint
    else:
        best_val_epoch = -1
        starting_epoch = 0
        if uda_train_only_discriminator <= starting_epoch:
            optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=lr)
        elif train_uda:
            optimizer = torch.optim.Adam(segmentation_model.discriminator.parameters(), lr=lr)
            for p in segmentation_model.unet.parameters():
                p.requires_grad = False
        train_loss = []
        val_loss = []
        adv_loss_per_epoch = []
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, num_epochs_scheduler, gamma=gamma_scheduler, last_epoch=starting_epoch - 1, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters=num_epochs, last_epoch=starting_epoch - 1, verbose=True)
    segmentation_model = load_model_by_gpu_ids(segmentation_model, gpu_ids)
    onnx_ok = None
    optimizer.zero_grad()
    it_number = 0
    for epoch in range(starting_epoch, num_epochs):
        if uda_train_only_discriminator == epoch and train_uda:
            optimizer = torch.optim.Adam(remove_data_parallel_warpping(segmentation_model).parameters(), lr=lr)
            remove_data_parallel_warpping(segmentation_model).unet.train()
            for p in remove_data_parallel_warpping(segmentation_model).unet.parameters():
                p.requires_grad = True
            #remove_data_parallel_warpping(segmentation_model).discriminator.eval()
            #for p in remove_data_parallel_warpping(segmentation_model).discriminator.parameters():
            #    p.requires_grad = False
        elif uda_train_only_discriminator > epoch and train_uda:
            remove_data_parallel_warpping(segmentation_model).discriminator.train()
            remove_data_parallel_warpping(segmentation_model).unet.eval()
            for p in remove_data_parallel_warpping(segmentation_model).unet.parameters():
                p.requires_grad = False
        elif train_uda:
            remove_data_parallel_warpping(segmentation_model).unet.train()
        gc.collect()
        loss_ep = 0.0
        adv_loss_ep = 0.0
        num_batches = 0
        if train_uda:
            train_unlabeled_dataloader_iter = iter(train_unlabeled_dataloader)
            adv_loss_func = nn.CrossEntropyLoss()
        for images, targets, paths in tqdm(dataloader):
            num_batches += 1
            images, targets = images.to(device), targets.to(device)
            if not train_uda:
                preds = segmentation_model(images)
                loss = loss_func(preds, targets)
                adv_loss = 0
            else:
                try:
                    unlabeled_images, unlabeled_paths, _ = next(train_unlabeled_dataloader_iter)
                except StopIteration:
                    train_unlabeled_dataloader_iter = iter(train_unlabeled_dataloader)
                    unlabeled_images, unlabeled_paths, _ = next(train_unlabeled_dataloader_iter)
                unlabeled_images = unlabeled_images.to(device)
                res = segmentation_model(images, unlabeled_images)
                if epoch >= uda_train_only_discriminator:
                    labeled_loss = loss_func(res['unet_labeled_preds'], targets)
                else:
                    labeled_loss = 0.0
                adv_loss = (0.5 * (res['discriminator_labeled_preds']) ** 2 +
                             0.5 * (res['discriminator_unlabeled_preds'] - 1) ** 2).mean()
                #adv_pred = torch.concat([res['discriminator_labeled_preds'], res['discriminator_unlabeled_preds']], dim=0)
                #adv_target = torch.tensor(([0] * res['discriminator_labeled_preds'].shape[0]) + ([1] * res['discriminator_unlabeled_preds'].shape[0])).to(device)
                #adv_loss = adv_loss_func(adv_pred, adv_target)
                loss = labeled_loss + lambda_uda * adv_loss
            loss.backward()
            nn.utils.clip_grad_norm_(segmentation_model.parameters(), 1)
            it_number += 1
            if it_number % accumulate == 0:
                optimizer.step()
                loss_ep += loss.item()
                if train_uda:
                    adv_loss_ep += adv_loss.item()
                optimizer.zero_grad()
        train_loss.append(loss_ep / float(num_batches))
        if not train_uda:
            val_loss.append(val(val_dataset, segmentation_model, loss_func, binary_mask_threshold, val_dataloader, validation_example_folder, classes_colors, device))
        else:
            val_loss.append(val_adv(epoch, uda_train_only_discriminator, lambda_uda, train_unlabeled_dataloader,
                                    val_dataset, segmentation_model, loss_func, binary_mask_threshold, val_dataloader, validation_example_folder, classes_colors, device))
        adv_loss_per_epoch.append(adv_loss_ep / float(num_batches))
        save_pytorch_model(segmentation_model, os.path.join(running_folder, 'last.pt'))
        if min(val_loss) == val_loss[-1] or (train_uda and len(val_loss) > uda_train_only_discriminator and min(val_loss[uda_train_only_discriminator:]) == val_loss[-1]):
            save_pytorch_model(segmentation_model, os.path.join(running_folder, 'segmentation.pt'))
            # save_onnx_model(torch.randn(images.shape, requires_grad=True, dtype=images.dtype).to(device), segmentation_model, os.path.join(running_folder, 'segmentation.onnx'))
            # onnx_ok = compare_onnx_and_pytorch_models(load_model_by_gpu_ids(os.path.join(running_folder, 'segmentation.onnx')), segmentation_model, device, images, random=True, decimal_point=4, log=log)
            best_val_epoch = epoch
        duration = time.time() - start_time
        running_data = {
            'running_name': running_name,
            'train_datasets_names': train_datasets_names,
            'val_datasets_names': val_datasets_names,
            'datasets_views_folder': datasets_views_folder,
            'datasets_folder': datasets_folder,
            'models_folder': models_folder,
            'detection_data_classes_field_names': detection_data_classes_field_names,
            'detection_data_fields_to_mask_in_loss': detection_data_fields_to_mask_in_loss,
            'rectangular_bounding_boxes': rectangular_bounding_boxes,
            'weight_by_class_num_pixels': weight_by_class_num_pixels,
            'binary_mask_threshold': binary_mask_threshold,
            'classes_colors': classes_colors,
            'num_epochs': num_epochs,
            'current_epoch': epoch,
            'best_val_epoch': best_val_epoch,
            'batch_size': batch_size,
            'lr': lr,
            'crop': crop,
            'depth': depth,
            'duration': duration,
            'num_parameters': segmentation_model_num_parameters,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'gpu_ids': gpu_ids,
            'enable_icon_or_emoji_text': enable_icon_or_emoji_text,
            'num_workers': num_workers,
            'random_transformation': random_transformation,
            'gamma_scheduler': gamma_scheduler,
            'num_epochs_scheduler': num_epochs_scheduler,
            'model_state_dict': get_model_state_dict(segmentation_model),
            'optimizer_state_dict': optimizer.state_dict(),
            'adv_loss_per_epoch': adv_loss_per_epoch,
            'uda_train_only_discriminator': uda_train_only_discriminator,
            'lambda_uda': lambda_uda
            # 'onnx_ok': onnx_ok
        }
        torch.save(running_data, os.path.join(running_folder, 'running_data.pt'))
        # json_object = json.dumps(running_data, indent=4)
        # with open(os.path.join(running_folder, 'running_data.json'), mode='w', encoding='utf-8') as data_file:
        #     data_file.write(json_object)
        log.info('Epoch : {}, Train : {}, Val : {}'.format(epoch, train_loss[-1], val_loss[-1]))
        lr_scheduler.step()
    return running_data


def val(dataset, segmentation_model, loss_func, binary_mask_threshold, dataloader: DataLoader, example_folder, colors, device):
    segmentation_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        num_batches = 0
        for images, targets, paths in tqdm(dataloader):
            num_batches += 1
            images, targets = images.to(device), targets.to(device)
            preds = segmentation_model(images)
            loss = loss_func(preds, targets)
            test_loss += loss.item()
        preds_to_pred_masks(preds, targets, paths, colors, example_folder, binary_mask_threshold)
        test_loss = test_loss / float(num_batches)
    return test_loss


def val_adv(epoch, uda_train_only_discriminator, lambda_uda, train_unlabeled_dataloader,
            dataset, segmentation_model, loss_func, binary_mask_threshold, dataloader: DataLoader, example_folder, colors, device):
    segmentation_model.eval()
    adv_loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        test_loss = 0.0
        num_batches = 0
        train_unlabeled_dataloader_iter = iter(train_unlabeled_dataloader)
        for images, targets, paths in tqdm(dataloader):
            num_batches += 1
            images, targets = images.to(device), targets.to(device)
            try:
                unlabeled_images, unlabeled_paths, _ = next(train_unlabeled_dataloader_iter)
            except StopIteration:
                train_unlabeled_dataloader_iter = iter(train_unlabeled_dataloader)
                unlabeled_images, unlabeled_paths, _ = next(train_unlabeled_dataloader_iter)
            unlabeled_images = unlabeled_images.to(device)
            res = segmentation_model(images, unlabeled_images)
            if epoch >= uda_train_only_discriminator:
                labeled_loss = loss_func(res['unet_labeled_preds'], targets)
                adv_loss = 0
            else:
                labeled_loss = 0.0
                adv_loss = (0.5 * (res['discriminator_labeled_preds']) ** 2 +
                                                     0.5 * (res['discriminator_unlabeled_preds'] - 1) ** 2).mean()
                adv_pred = torch.concat([res['discriminator_labeled_preds'], res['discriminator_unlabeled_preds']], dim=0)
                adv_target = torch.tensor(([0] * res['discriminator_labeled_preds'].shape[0]) + (
                            [1] * res['discriminator_unlabeled_preds'].shape[0])).to(device)
                adv_loss = adv_loss_func(adv_pred, adv_target)
            #adv_loss = 0
            loss = labeled_loss + lambda_uda * adv_loss
            test_loss += loss.item()
        preds_to_pred_masks(res['unet_labeled_preds'], targets, paths, colors, example_folder, binary_mask_threshold)
        test_loss = test_loss / float(num_batches)
    return test_loss


def preds_to_pred_masks(preds, targets, paths, colors, dst_folder, binary_mask_threshold=0.5):
    output = (preds.detach().cpu()[:, 1, :, :] >= binary_mask_threshold) * 1
    for isample, sample in enumerate(range(len(paths))):
        color_image = torch.zeros((3, output.shape[1], output.shape[2])).long()
        sample_output = output[sample, :, :]
        if targets is not None:
            color_image_target = torch.zeros((3, output.shape[1], output.shape[2])).long()
            sample_target = targets[sample, :, :]
        for i in range(len(colors)):
            color_image[:, sample_output == i] = colors[i]
            if targets is not None:
                color_image_target[:, sample_target == i] = colors[i]
        Image.fromarray(color_image.permute((1, 2, 0)).numpy().astype(np.uint8)).convert('RGB').save(os.path.join(dst_folder, paths[sample].split(os.sep)[-1]), compress_level=0)
        if targets is not None:
            Image.fromarray(color_image_target.permute((1, 2, 0)).numpy().astype(np.uint8)).convert('RGB').save(os.path.join(dst_folder, paths[sample].split(os.sep)[-1] + '_target.png'), compress_level=0)

