import torch
import torch.nn as nn
import copy
from quantization.qat.qat_layers import Conv1dEncoderQ, ConvTr1dDecoderQ, Conv2dEncoderQ, LinearDecoderQ, ConvTr2dDecoderQ
from quantization.qat.qat_layers import Add, Sub, Mul, Div, Const
from quantization.qat.qat_layers import Conv1dQ, Conv2dQ, ConvTranspose1dQ, ConvTranspose2dQ, Conv1dGnNlQ
from quantization.qat.qat_layers import Conv1dNlQ, Conv2dNlQ, ConvTranspose1dNlQ, ConvTranspose2dNlQ
from quantization.qat.qat_layers import NlQ, GroupNormQ, AddQ, SubQ, MulQ, DivQ, ConstQ
from quantization.qat.qat_layers import LSTMQ, MultiheadAttentionQ, LinearQ, LinearNlQ, LayerNormQ, EmbeddingQ, BatchNormQ
from quantization.qat.qat_quant import TorchActivationFakeQuantize, TorchWeightFakeQuantize, TorchDymActivationFakeQuantize

def quant_encoderq(encoder, params_dict):
    if isinstance(encoder[0],nn.Conv1d):
        return Conv1dEncoderQ(encoder,
                              n_splitter=params_dict.get('n_splitter'),
                              gradient_based=params_dict.get('gradient_based'),
                              act_quant=params_dict.get('act_quant'),
                              inout_nl_quant=params_dict.get('inout_nl_quant'),
                              weight_quant=params_dict.get('weight_quant'),
                              in_quant=params_dict.get('in_quant'),
                              act_n_bits=params_dict.get('act_n_bits'),
                              weight_n_bits=params_dict.get('weight_n_bits'),
                              in_act_n_bits=params_dict.get('in_act_n_bits'))
    elif isinstance(encoder[0],nn.Conv2d):
        return Conv2dEncoderQ(encoder,
                              n_splitter=params_dict.get('n_splitter'),
                              gradient_based=params_dict.get('gradient_based'),
                              act_quant=params_dict.get('act_quant'),
                              inout_nl_quant=params_dict.get('inout_nl_quant'),
                              weight_quant=params_dict.get('weight_quant'),
                              in_quant=params_dict.get('in_quant'),
                              act_n_bits=params_dict.get('act_n_bits'),
                              weight_n_bits=params_dict.get('weight_n_bits'),
                              in_act_n_bits=params_dict.get('in_act_n_bits'))
    else:
        assert False, "No support!"

def quant_decoderq(decoder, params_dict):
    if isinstance(decoder[0],nn.ConvTranspose1d):
        return ConvTr1dDecoderQ(decoder,
                                n_combiner=params_dict.get('n_combiner'),
                                gradient_based=params_dict.get('gradient_based'),
                                act_quant=params_dict.get('act_quant'),
                                inout_nl_quant=params_dict.get('inout_nl_quant'),
                                weight_quant=params_dict.get('weight_quant'),
                                weight_n_bits=params_dict.get('weight_n_bits'),
                                act_n_bits=params_dict.get('act_n_bits'),
                                out_quant=params_dict.get('out_quant'),
                                out_act_n_bits=params_dict.get('out_act_n_bits'),
                                train_res_dec=params_dict.get('train_res_dec'))
    elif isinstance(decoder[0],nn.ConvTranspose2d):
        return ConvTr2dDecoderQ(decoder,
                                n_combiner=params_dict.get('n_combiner'),
                                gradient_based=params_dict.get('gradient_based'),
                                act_quant=params_dict.get('act_quant'),
                                inout_nl_quant=params_dict.get('inout_nl_quant'),
                                weight_quant=params_dict.get('weight_quant'),
                                weight_n_bits=params_dict.get('weight_n_bits'),
                                act_n_bits=params_dict.get('act_n_bits'),
                                out_quant=params_dict.get('out_quant'),
                                out_act_n_bits=params_dict.get('out_act_n_bits'),
                                train_res_dec=params_dict.get('train_res_dec'))
    elif isinstance(decoder[0],nn.Linear):
        return LinearDecoderQ(decoder,
                            n_combiner=params_dict.get('n_combiner'),
                            gradient_based=params_dict.get('gradient_based'),
                            act_quant=params_dict.get('act_quant'),
                            inout_nl_quant=params_dict.get('inout_nl_quant'),
                            weight_quant=params_dict.get('weight_quant'),
                            weight_n_bits=params_dict.get('weight_n_bits'),
                            act_n_bits=params_dict.get('act_n_bits'),
                            out_quant=params_dict.get('out_quant'),
                            out_act_n_bits=params_dict.get('out_act_n_bits'),
                            train_res_dec=params_dict.get('train_res_dec'))
    else:
        assert False, "No support!"

def quant_conv1d(conv1d, params_dict):
    return Conv1dQ(conv1d,
                   gradient_based=params_dict.get('gradient_based'),
                   act_quant=params_dict.get('act_quant'),
                   weight_quant=params_dict.get('weight_quant'),
                   act_n_bits=params_dict.get('act_n_bits'),
                   weight_n_bits=params_dict.get('weight_n_bits'))

def quant_conv2d(conv2d, params_dict):
    return Conv2dQ(conv2d,
                   gradient_based=params_dict.get('gradient_based'),
                   act_quant=params_dict.get('act_quant'),
                   weight_quant=params_dict.get('weight_quant'),
                   act_n_bits=params_dict.get('act_n_bits'),
                   weight_n_bits=params_dict.get('weight_n_bits'))

def quant_convtr1d(convTr1d, params_dict):
    return ConvTranspose1dQ(convTr1d,
                            gradient_based=params_dict.get('gradient_based'),
                            act_quant=params_dict.get('act_quant'),
                            weight_quant=params_dict.get('weight_quant'),
                            act_n_bits=params_dict.get('act_n_bits'),
                            weight_n_bits=params_dict.get('weight_n_bits'))

def quant_convtr2d(convTr2d, params_dict):
    return ConvTranspose2dQ(convTr2d,
                            gradient_based=params_dict.get('gradient_based'),
                            act_quant=params_dict.get('act_quant'),
                            weight_quant=params_dict.get('weight_quant'),
                            act_n_bits=params_dict.get('act_n_bits'),
                            weight_n_bits=params_dict.get('weight_n_bits'))

def quant_conv1d_nl(conv1d, nl, params_dict):
    return Conv1dNlQ(conv1d, nl,
                     gradient_based=params_dict.get('gradient_based'),
                     act_quant=params_dict.get('act_quant'),
                     weight_quant=params_dict.get('weight_quant'),
                     act_n_bits=params_dict.get('act_n_bits'),
                     weight_n_bits=params_dict.get('weight_n_bits'))

def quant_conv1d_gn_nl(conv1d, gn, nl, params_dict):
    return Conv1dGnNlQ(conv1d, gn, nl,
                     gradient_based=params_dict.get('gradient_based'),
                     act_quant=params_dict.get('act_quant'),
                     weight_quant=params_dict.get('weight_quant'),
                     act_n_bits=params_dict.get('act_n_bits'),
                     weight_n_bits=params_dict.get('weight_n_bits'))

def quant_conv2d_nl(conv2d, nl, params_dict):
    return Conv2dNlQ(conv2d, nl,
                     gradient_based=params_dict.get('gradient_based'),
                     act_quant=params_dict.get('act_quant'),
                     weight_quant=params_dict.get('weight_quant'),
                     act_n_bits=params_dict.get('act_n_bits'),
                     weight_n_bits=params_dict.get('weight_n_bits'))

def quant_convtr1d_nl(convTr1d, nl, params_dict):
    return ConvTranspose1dNlQ(convTr1d, nl,
                             gradient_based=params_dict.get('gradient_based'),
                             act_quant=params_dict.get('act_quant'),
                             weight_quant=params_dict.get('weight_quant'),
                             act_n_bits=params_dict.get('act_n_bits'),
                             weight_n_bits=params_dict.get('weight_n_bits'))

def quant_convtr2d_nl(convTr2d, nl, params_dict):
    return ConvTranspose2dNlQ(convTr2d, nl,
                             gradient_based=params_dict.get('gradient_based'),
                             act_quant=params_dict.get('act_quant'),
                             weight_quant=params_dict.get('weight_quant'),
                             act_n_bits=params_dict.get('act_n_bits'),
                             weight_n_bits=params_dict.get('weight_n_bits'))

def quant_groupnorm(groupnorm, params_dict):
    return GroupNormQ(groupnorm,
                      gradient_based=params_dict.get('gradient_based'),
                      act_quant=params_dict.get('act_quant'),
                      act_n_bits=params_dict.get('act_n_bits'))

def quant_layernorm(layernorm, params_dict):
    return LayerNormQ(layernorm,
                      gradient_based=params_dict.get('gradient_based'),
                      act_quant=params_dict.get('act_quant'),
                      act_n_bits=params_dict.get('act_n_bits'))

def quant_batchnorm(batchnorm, params_dict):
    return BatchNormQ(batchnorm,
                      gradient_based=params_dict.get('gradient_based'),
                      act_quant=params_dict.get('act_quant'),
                      act_n_bits=params_dict.get('act_n_bits'))

def quant_embedding(embedding, params_dict):
    return EmbeddingQ(embedding,
                      gradient_based=params_dict.get('gradient_based'),
                      act_quant=params_dict.get('act_quant'),
                      weight_quant=params_dict.get('weight_quant'),
                      act_n_bits=params_dict.get('act_n_bits'),
                      weight_n_bits=params_dict.get('weight_n_bits'))

def quant_nl(nl, params_dict):
    return NlQ(nl,
               gradient_based=params_dict.get('gradient_based'),
               act_quant=params_dict.get('act_quant'),
               act_n_bits=params_dict.get('act_n_bits'))

def quant_linear(linear, params_dict):
    return LinearQ(linear,
                   gradient_based=params_dict.get('gradient_based'),
                   act_quant=params_dict.get('act_quant'),
                   weight_quant=params_dict.get('weight_quant'),
                   act_n_bits=params_dict.get('act_n_bits'),
                   weight_n_bits=params_dict.get('weight_n_bits'))

def quant_linear_nl(linear, nl, params_dict):
    return LinearNlQ(linear,
                    nl,
                    gradient_based=params_dict.get('gradient_based'),
                    act_quant=params_dict.get('act_quant'),
                    weight_quant=params_dict.get('weight_quant'),
                    act_n_bits=params_dict.get('act_n_bits'),
                    weight_n_bits=params_dict.get('weight_n_bits'))

def quant_mha(mha, params_dict):
    return MultiheadAttentionQ(mha,
                               gradient_based=params_dict.get('gradient_based'),
                               act_quant=params_dict.get('act_quant'),
                               weight_quant=params_dict.get('weight_quant'),
                               act_n_bits=params_dict.get('act_n_bits'),
                               weight_n_bits=params_dict.get('weight_n_bits'))

def quant_lstm(lstm, params_dict):
    return LSTMQ(lstm,
                 gradient_based=params_dict.get('gradient_based'),
                 act_quant=params_dict.get('act_quant'),
                 weight_quant=params_dict.get('weight_quant'),
                 act_n_bits=params_dict.get('act_n_bits'),
                 weight_n_bits=params_dict.get('weight_n_bits'))

def quant_add(add, params_dict):
    return AddQ(add,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))

def quant_sub(sub, params_dict):
    return SubQ(sub,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))


def quant_mul(mul, params_dict):
    return MulQ(mul,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))

def quant_div(div, params_dict):
    return DivQ(div,
                gradient_based=params_dict.get('gradient_based'),
                act_quant=params_dict.get('act_quant'),
                act_n_bits=params_dict.get('act_n_bits'))

def quant_const(const, params_dict):
    return ConstQ(const,
                  gradient_based=params_dict.get('gradient_based'),
                  act_quant=params_dict.get('act_quant'),
                  act_n_bits=params_dict.get('act_n_bits'))

def torch_weight_quantizer(quantizer):
    return TorchWeightFakeQuantize(quantizer)


def torch_activation_quantizer(quantizer):
    return TorchActivationFakeQuantize(quantizer)


def torch_dym_activation_quantizer(quantizer):
    return TorchDymActivationFakeQuantize(quantizer)


def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def quantize_known_modules(mod_list, params_dict):
    types = tuple(type(m) for m in mod_list)
    if len(types)==1:
        types = types[0]
    quantize_method = OP_LIST_TO_QUANTIZE_METHOD.get(types, None)
    if quantize_method is None:
        raise NotImplementedError("Cannot quantize modules: {}".format(types))
    new_mod = [None] * len(mod_list)
    new_mod[0] = quantize_method(*mod_list, params_dict)

    for i in range(1, len(mod_list)):
        new_mod[i] = torch.nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod

def _quantize_modules(model, modules_to_quantize, params_dict, replacer_func=quantize_known_modules):
    mod_list = []
    for item in modules_to_quantize:
        mod_list.append(_get_module(model, item))

    # Quantize list of modules
    new_mod_list = replacer_func(mod_list, params_dict)

    # Replace original module list with quantize module list
    for i, item in enumerate(modules_to_quantize):
        _set_module(model, item, new_mod_list[i])

def quantize_modules(model,
                     modules_to_quantize,
                     params_dict={},
                     inplace=True,
                     replacer_func=quantize_known_modules):
    if not inplace:
        model = copy.deepcopy(model)
    # Handle case of modules_to_fuse being a list
    _quantize_modules(model, modules_to_quantize, params_dict, replacer_func)
    return model

def replace_encoderq(model, modules_to_replace, params_dict):
    mod_list = []
    for item in modules_to_replace:
        mod_list.append(_get_module(model, item))
    # Replace list of modules
    new_mod = quant_encoderq(mod_list, params_dict)
    # Replace original module list with quantize module list
    _set_module(model, modules_to_replace[0], new_mod)
    for i in range(1,len(modules_to_replace)):
        _set_module(model, modules_to_replace[i], nn.Identity())

def replace_decoderq(model, modules_to_replace, params_dict):
    mod_list = []
    for item in modules_to_replace:
        mod_list.append(_get_module(model, item))
    # Replace list of modules
    new_mod = quant_decoderq(mod_list, params_dict)
    # Replace original module list with quantize module list
    _set_module(model, modules_to_replace[0], new_mod)
    for i in range(1,len(modules_to_replace)):
        _set_module(model, modules_to_replace[i], nn.Identity())

def replace_weight_quantizer(model, module_to_replace, module):
    # Create new module
    new_module = torch_weight_quantizer(module)
    # Replace original module list with new module
    _set_module(model, module_to_replace, new_module)


def replace_activation_quantizer(model, module_to_replace, module):
    # Create new module
    new_module = torch_activation_quantizer(module)
    # Replace original module list with new module
    _set_module(model, module_to_replace, new_module)

def replace_dym_activation_quantizer(model, module_to_replace, module):
    # Create new module
    new_module = torch_dym_activation_quantizer(module)
    # Replace original module list with new module
    _set_module(model, module_to_replace, new_module)


OP_LIST_TO_QUANTIZE_METHOD = {
    (nn.Conv1d): quant_conv1d,
    (nn.Conv2d): quant_conv2d,
    (nn.ConvTranspose1d): quant_convtr1d,
    (nn.ConvTranspose2d): quant_convtr2d,
    (nn.Conv1d, nn.PReLU): quant_conv1d_nl,
    (nn.Conv1d, nn.ReLU): quant_conv1d_nl,
    (nn.Conv1d, nn.Tanh): quant_conv1d_nl,
    (nn.Conv1d, nn.Sigmoid): quant_conv1d_nl,
    (nn.Conv1d, nn.GELU): quant_conv1d_nl,
    (nn.Conv1d, nn.GLU): quant_conv1d_nl,
    (nn.Conv1d, nn.GroupNorm, nn.PReLU): quant_conv1d_gn_nl,
    (nn.Conv1d, nn.GroupNorm, nn.ReLU): quant_conv1d_gn_nl,
    (nn.Conv1d, nn.GroupNorm, nn.Tanh): quant_conv1d_gn_nl,
    (nn.Conv1d, nn.GroupNorm, nn.Sigmoid): quant_conv1d_gn_nl,
    (nn.Conv1d, nn.GroupNorm, nn.GELU): quant_conv1d_gn_nl,
    (nn.Conv1d, nn.GroupNorm, nn.GLU): quant_conv1d_gn_nl,
    (nn.Conv2d, nn.PReLU): quant_conv2d_nl,
    (nn.Conv2d, nn.ReLU): quant_conv2d_nl,
    (nn.Conv2d, nn.Tanh): quant_conv2d_nl,
    (nn.Conv2d, nn.Sigmoid): quant_conv2d_nl,
    (nn.Conv2d, nn.GELU): quant_conv2d_nl,
    (nn.Conv2d, nn.GLU): quant_conv2d_nl,
    (nn.ConvTranspose1d, nn.GELU): quant_convtr1d_nl,
    (nn.ConvTranspose2d, nn.GELU): quant_convtr2d_nl,
    (nn.GroupNorm): quant_groupnorm,
    (nn.LayerNorm): quant_layernorm,
    (nn.BatchNorm1d): quant_batchnorm,
    (nn.BatchNorm2d): quant_batchnorm,
    (nn.Embedding): quant_embedding,
    (nn.PReLU): quant_nl,
    (nn.ReLU): quant_nl,
    (nn.LeakyReLU): quant_nl,
    (nn.Sigmoid): quant_nl,
    (nn.Tanh): quant_nl,
    (nn.GELU): quant_nl,
    (nn.GLU): quant_nl,
    (nn.LSTM): quant_lstm,
    (nn.MultiheadAttention): quant_mha,
    (nn.Linear): quant_linear,
    (nn.Linear, nn.ReLU): quant_linear_nl,
    (nn.Linear, nn.GELU): quant_linear_nl,
    (Add): quant_add,
    (Sub): quant_sub,
    (Mul): quant_mul,
    (Div): quant_div,
    (Const): quant_const,
}
