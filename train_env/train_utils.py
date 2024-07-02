import copy
import torch
import numpy as np
from process import generate_2mix_snr, generate_3mix_snr
from quantization.qat.models.load_model import create_model, quantize_model


def create_pretrained_model(model_cfg, use_weights=True):
    # Create model
    model = create_model(model_cfg)
    model_path = model_cfg['model_path']
    if use_weights and model_path is not None:
        if model_path.startswith("https"):
            model_state_dict = torch.hub.load_state_dict_from_url(model_path, map_location='cpu', check_hash=True)
        else:
            model_state_dict = torch.load(model_path)
        try:
            model.load_state_dict(model_state_dict.get("state",model_state_dict), strict=True)
        except:
            try:
                model.load_pretrain(model_path)
            except:
                print("Warning: No pretraind weights were loaded!")
    # Quantize model
    fmodel = copy.deepcopy(model)
    model = quantize_model(model, model_cfg['quantization'])
    return model, fmodel


def augmentation_2mix(signal1, signal2, augmentation_cfg):
    if augmentation_cfg.get('distribution') == "uniform":
        min_snr = augmentation_cfg.get('param0')
        max_snr = augmentation_cfg.get('param1')
        # Random uniform snr
        snr = np.random.uniform(low=min_snr, high=max_snr)
        mixture = generate_2mix_snr(signal1, signal2, snr)
    else:
        assert False, "Augmentation is not supoorted!"
    return mixture


def augmentation_3mix(signal1, signal2, signal3, augmentation_cfg):
    if augmentation_cfg.get('distribution') == "uniform":
        min_snr = augmentation_cfg.get('param0')
        max_snr = augmentation_cfg.get('param1')
        # Random uniform snr
        snr1_23 = np.random.uniform(low=min_snr, high=max_snr)
        snr2_3 = np.random.uniform(low=min_snr, high=max_snr)
        mixture = generate_3mix_snr(signal1, signal2, signal3, snr1_23, snr2_3)
    else:
        assert False, "Augmentation is not supoorted!"
    return mixture

