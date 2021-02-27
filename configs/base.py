import base64
import logging
import argparse
import json
from argparse import ArgumentParser

import nni

from utils import prepare_experiment


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(v):
    lst = json.loads(v)
    if not isinstance(lst, list):
        raise argparse.ArgumentTypeError('Dumped string of list expected.')
    return lst


def str2dict(v):
    obj = json.loads(v)
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError('Dumped string of dict expected.')
    return obj


def comma_separated_list(v):
    return list(filter(lambda d: d, v.split(",")))


class BasicParser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.defaults = dict()
        for k, v in self.default_params().items():
            if not isinstance(v, dict):
                if isinstance(v, list):
                    v = {"default": v[0], "choices": v}
                else:
                    v = {"default": v, "type": self._infer_type(v)}
            if "type" not in v:
                v["type"] = self._infer_type(v["default"])
            if "action" in v:
                v.pop("type")
            if "default" in v:
                self.defaults[k] = v["default"]
            self.add_argument("--" + k, **v)

    def _infer_type(self, inst):
        if isinstance(inst, bool):
            return str2bool
        if inst is None:
            return str
        return type(inst)

    def default_params(self):
        return {
            "debug": {"default": False, "action": "store_true"},
            "seed": 42,
            "output_dir": "outputs",
            "tmp_dir": "tmp",
            # this should be a shared folder among trials, which will be converted into trial specific folder in preparation
            "log_frequency": 20,
            "num_threads": 4,
            "base64_params": "",
            "fast_forward": True,
        }

    def validate_args(self, args):
        pass

    def parse_args(self, args=None, namespace=None):
        namespace = super().parse_args(args=args, namespace=namespace)
        params = nni.get_next_parameter()
        if not params:
            params = {}
        if namespace.base64_params:
            params.update(json.loads(base64.b64decode(namespace.base64_params).decode()))
        if params:
            for k, v in params.items():
                assert hasattr(namespace, k), "Args doesn't have received key: %s" % k
                setattr(namespace, k, v)

        self.validate_args(namespace)
        prepare_experiment(namespace)
        return namespace

    @classmethod
    def parse_configs(cls):
        return cls().parse_args()
