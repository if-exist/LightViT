import yaml
import torch
import logging

from .nas_model import gen_nas_model
from .darts_model import gen_darts_model
from .mobilenet_v1 import MobileNetV1
from . import resnet, lightvit_v1

logger = logging.getLogger()


def build_model(args, model_name, pretrained=False, pretrained_ckpt=''):
    if model_name.lower().startswith('lightvit'):
        from . import lightvit
        model = getattr(lightvit, model_name)(num_classes=200, distillation=False)
    else:
        raise RuntimeError(f'Model {model_name} not found.')

    if pretrained and pretrained_ckpt != '':
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = \
                model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) != 0:
            logger.info(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.info(f'Unexpected keys in source state dict: {unexpected_keys}')

    return model


def build_edgenn_model(args, edgenn_cfgs=None):
    import edgenn
    if args.model.lower() in ['nas_model', 'nas_pruning_model']:
        # gen model with yaml config first
        model = gen_nas_model(yaml.load(open(args.model_config, 'r'), Loader=yaml.FullLoader), drop_rate=args.drop, drop_path_rate=args.drop_path_rate)
        # wrap the model with EdgeNNModel
        model = edgenn.models.EdgeNNModel(model, loss_fn, pruning=(args.model=='nas_pruning_model'))

    elif args.model == 'edgenn':
        # build model from edgenn
        model = edgenn.build_model(edgenn_cfgs.model)

    else:
        raise RuntimeError(f'Model {args.model} not found.')

    return model
