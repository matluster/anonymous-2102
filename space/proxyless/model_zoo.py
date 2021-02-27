import json

import torch

from .network import ProxylessNASNet


def proxyless_base(pretrained=True, net_config=None, net_weight=None):
    assert net_config is not None, "Please input a network config"
    with open(net_config, "r") as f:
        net_config_json = json.load(f)
    net = ProxylessNASNet.from_config(net_config_json)

    if "bn" in net_config_json:
        net.set_bn_param(
            momentum=net_config_json["bn"]["momentum"],
            eps=net_config_json["bn"]["eps"])
    else:
        net.set_bn_param(momentum=0.1, eps=1e-3)

    if pretrained:
        assert net_weight is not None, "Please specify network weights"
        init = torch.load(net_weight, map_location="cpu")
        net.load_state_dict(init["state_dict"])

    return net


def proxyless_cpu():
    return proxyless_base(net_config="data/proxyless/proxyless_cpu.config",
                          net_weight="data/proxyless/proxyless_cpu.pth")


def proxyless_gpu():
    return proxyless_base(net_config="data/proxyless/proxyless_gpu.config",
                          net_weight="data/proxyless/proxyless_gpu.pth")


def proxyless_mobile_14():
    return proxyless_base(net_config="data/proxyless/proxyless_mobile_14.config",
                          net_weight="data/proxyless/proxyless_mobile_14.pth")


def proxyless_mobile():
    return proxyless_base(net_config="data/proxyless/proxyless_mobile.config",
                          net_weight="data/proxyless/proxyless_mobile.pth")
