import torch
from quantization.qat.models.convtasnetq import ConvTasNetQ
from quantization.qat.models.convtasnetq_music import ConvTasNetMusicQ
from quantization.qat.models.dptnetq import DPTNetQ
from quantization.qat.models.sepformerq import SepformerQ
from quantization.qat.models.htdemucsq import HTDemucsQ
from quantization.qat.qat_quant import GradientActivationFakeQuantize, GradientWeightFakeQuantize
from quantization.qat.qat_layers import LayerQ


def set_mac_op(model, mode=False):
    for _, m in model.named_modules():
        if isinstance(m, LayerQ):
            m.do_mac_op = mode

def enable_observer(model, mode=False):
    for _, m in model.named_modules():
        if isinstance(m, GradientWeightFakeQuantize) or isinstance(m, GradientActivationFakeQuantize):
            m.enable_observer(mode)

def create_model(model_cfg):
    name = model_cfg['name']
    if name == "ConvTasNet":
        model = ConvTasNetQ(n_spks=model_cfg.get('n_src',1),
                            kernel_size=model_cfg.get('kernel_size',32),
                            stride=model_cfg.get('stride',16))
    elif name == "DPTNet":
        model = DPTNetQ(n_spks=model_cfg.get('n_src',2),
                        kernel_size=model_cfg.get('kernel_size',2))
    elif name == "Sepformer":
        model = SepformerQ(n_spks=model_cfg.get('n_src',2),
                           kernel_size=model_cfg.get('kernel_size', 16),
                           stride=model_cfg.get('stride', 8))
    elif name == "ConvTasNetMusic":
        model = ConvTasNetMusicQ(sources=model_cfg.get('sources',['drums', 'bass', 'other', 'vocals']),
                                 kernel=model_cfg.get('kernel_size', 20),
                                 stride=model_cfg.get('stride', 10))
    elif name == "HTDemucs":
        model_path = model_cfg.get('model_path',None)
        if model_path:
            if model_path.startswith("https"):
                model_state_dict = torch.hub.load_state_dict_from_url(model_path, map_location='cpu', check_hash=True)
            else:
                model_state_dict = torch.load(model_path)
            model = HTDemucsQ(**model_state_dict['kwargs'])
        else:
            model = HTDemucsQ(sources=model_cfg.get('sources',['drums','bass','other','vocals']))
    else:
        assert False, "Model {} is not supported!".format(name)

    return model

def quantize_model(model, quant_cfg):
    if quant_cfg.get('qat', False):
        gradient_based = quant_cfg.get('gradient_based', True)
        weight_quant, weight_n_bits = quant_cfg.get('weight_quant', True), quant_cfg.get('weight_n_bits', 8)
        act_quant, act_n_bits = quant_cfg.get('act_quant', True), quant_cfg.get('act_n_bits', 8)
        in_quant, in_act_n_bits = quant_cfg.get('in_quant', False), quant_cfg.get('in_act_n_bits', 8)
        out_quant, out_act_n_bits = quant_cfg.get('out_quant', False), quant_cfg.get('out_act_n_bits', 8)
        n_splitter, n_combiner = quant_cfg.get('n_splitter', 1), quant_cfg.get('n_combiner', 1)
        inout_nl_quant = quant_cfg.get('inout_nl_quant', False)
        model.set_splitter_combiner(n_splitter, n_combiner)
        model.quantize_model(gradient_based=gradient_based,
                             weight_quant=weight_quant,
                             weight_n_bits=weight_n_bits,
                             act_quant=act_quant,
                             act_n_bits=act_n_bits,
                             inout_nl_quant=inout_nl_quant,
                             in_quant=in_quant,
                             in_act_n_bits=in_act_n_bits,
                             out_quant=out_quant,
                             out_act_n_bits=out_act_n_bits)
        enable_observer(model, quant_cfg.get('observer', False))
    return model

def create_pretrained_model(model_cfg):
    # Create model
    model = create_model(model_cfg)
    # Quantize model
    model = quantize_model(model, model_cfg['quantization'])
    # Load model
    model_path = model_cfg.get('model_path', None)
    if model_path is None:
        return model
    elif model_path.startswith("https"):
        model_state_dict = torch.hub.load_state_dict_from_url(model_path, map_location='cpu', check_hash=True)
    else:
        model_state_dict = torch.load(model_path)
    try:
        if "state" in model_state_dict:
            model.load_state_dict(model_state_dict["state"], strict=True)
        elif "state_dict" in model_state_dict:
            model.load_state_dict(model_state_dict["state_dict"], strict=True)
        else:
            model.load_state_dict(model_state_dict, strict=True)
    except:
        try:
            model.load_pretrain(model_path)
        except:
            print("Error: mismatch models weights. Please check if the model configurations match to model weights!")
            exit()
    return model
