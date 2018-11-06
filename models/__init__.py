"""
Keras example model factory functions.
"""

def get_model(name, **model_args):
    if name == 'cnn':
        from .cnn import build_model
        return build_model(**model_args)
    elif name == 'resnet18_cifar':
        from .resnet import build_resnet18_cifar
        return build_resnet18_cifar(**model_args)
    elif name == 'resnet50':
        from .resnet import build_resnet50
        return build_resnet50(**model_args)
    elif name == 'resnet50_official':
        from .resnet_official import build_resnet50
        return build_resnet50(**model_args)
    else:
        raise ValueError('Model %s unknown' % name)
