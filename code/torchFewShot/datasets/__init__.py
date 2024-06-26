from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .miniImageNet import miniImageNet
from .tieredImageNet import tieredImageNet
from .miniImageNet_load import miniImageNet_load
from .tieredImageNet_load import tieredImageNet_load
from .CIFAR_FS_load import CIFAR_FS_load


__imgfewshot_factory = {
        'miniImageNet': miniImageNet,
        'tieredImageNet': tieredImageNet,
        'miniImageNet_load': miniImageNet_load,
        'tieredImageNet_load': tieredImageNet_load,
        'CIFAR_FS_load': CIFAR_FS_load,
}


def get_names():
    return list(__imgfewshot_factory.keys()) 


def init_imgfewshot_dataset(name, **kwargs):
    if name not in list(__imgfewshot_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgfewshot_factory.keys())))
    return __imgfewshot_factory[name](**kwargs)

