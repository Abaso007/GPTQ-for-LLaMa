import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res |= find_layers(
            child,
            layers=layers,
            name=f'{name}.{name1}' if name != '' else name1,
        )
    return res


def gen_conditions(_wbits, _groupsize):
    wbits = _wbits
    groupsize = _groupsize
    conditions = []
    while wbits < 8 or groupsize not in [-1, 32]:
        if groupsize > 32:
            groupsize /= 2
        else:
            wbits *= 2
            groupsize = _groupsize

        conditions.append((int(wbits), int(groupsize)))
    return conditions
