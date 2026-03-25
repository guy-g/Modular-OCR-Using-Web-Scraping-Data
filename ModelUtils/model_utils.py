import gc
import torch
import onnxruntime as ort
import numpy as np
import torch.nn as nn
from Recognition.model import *
from Detection.model import *


def get_device(gpu_ids):
    if gpu_ids == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu_ids.split(',')[0]))
    return device


def load_model_by_gpu_ids(model, gpu_ids=None, device=None):
    if gpu_ids is not None:
        device = get_device(gpu_ids)
    if type(model) == str and model.split('.')[-1] != 'onnx':
        model = torch.load(model, map_location=device, weights_only=False).eval()
    elif type(model) == str and model.split('.')[-1] == 'onnx':
        model = ort.InferenceSession(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    elif 'onnxruntime' not in str(model.__class__):
        model = remove_data_parallel_warpping(model)
        model = model.to(device).eval()
    if gpu_ids is not None and len(gpu_ids.split(',')) > 1 and 'onnxruntime' not in str(model.__class__):
        model = nn.DataParallel(model, device_ids=[int(gpu_id) for gpu_id in gpu_ids.split(',')]).eval()
    return model


def get_model_state_dict(model):
    inner_model = remove_data_parallel_warpping(model)
    return inner_model.state_dict()


def save_pytorch_model(model, file_path):
    model_to_save = remove_data_parallel_warpping(model)
    torch.save(model_to_save, file_path)


def remove_data_parallel_warpping(model):
    if model.__class__ == nn.DataParallel:
        inner_model = model.module
    else:
        inner_model = model
    return inner_model


def save_onnx_model(dummy_input, model, file_path, dynamic_width=False):
    inner_model = remove_data_parallel_warpping(model)
    if dynamic_width:
        dynamic_axes = {
            'input': {0: 'batch_size', 3: 'width'},
            'output': {0: 'batch_size', 3: 'width'}
        }
    else:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    torch.onnx.export(
        inner_model,
        dummy_input,
        file_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

def inference_model(model, x, keys=None):
    if 'onnxruntime' not in str(model.__class__):
        return model(x)
    return inference_onnx_model(model, x, keys)


def inference_onnx_model(onnx_model, x, keys=None):
    output = onnx_model.run(None, {'input': x.detach().cpu().numpy()})
    output = [torch.Tensor(i) for i in output]
    if len(output) == 1:
        return output[0]
    elif keys is not None:
        return {k: output[ik] for ik, k in enumerate(keys)}
    return output


def compare_onnx_and_pytorch_models(onnx_model, pytorch_model, pytorch_device, images, random=True, decimal_point=4, log=None):
    if random:
        pytorch_images = torch.randn(images.shape, dtype=images.dtype).to(pytorch_device)
    else:
        pytorch_images = images.to(pytorch_device)
    out_pytorch = pytorch_model(pytorch_images)
    out_onnx = inference_onnx_model(onnx_model, pytorch_images)
    try:
        if type(out_pytorch) == dict:
            for ik, k in enumerate(out_pytorch.keys()):
                np.testing.assert_almost_equal(out_pytorch[k].detach().cpu().numpy(), out_onnx[ik], decimal_point)
        else:
            np.testing.assert_almost_equal(out_pytorch.detach().cpu().numpy(), out_onnx, decimal_point)
        if log is not None:
            log.info('The onnx is good!')
        else:
            print('The onnx is good!')
        return True
    except Exception as e:
        if log is not None:
            log.warning(str(e))
        else:
            print(str(e))
        return False


def get_number_of_parameters(pytorch_model):
    trainable_params = sum(
        p.numel() for p in pytorch_model.parameters() if p.requires_grad
    )
    total_params = sum(
        param.numel() for param in pytorch_model.parameters()
    )
    return {
        'trainable_params': trainable_params,
        'total_params': total_params
    }


def clear_cache(log=None):
    gc.collect()
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        mem = 'Before clearing cache: total memory - {}, reserved memory - {}, allocated memory - {}'.format(t, r, a)
        if log is not None:
            log.info(mem)
        else:
            print(mem)
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        mem = 'After clearing cache: total memory - {}, reserved memory - {}, allocated memory - {}'.format(t, r, a)
        if log is not None:
            log.info(mem)
        else:
            print(mem)

