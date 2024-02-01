from typing import *
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np

from alamo.interface import AlamoInterface
from alamo.manager import AlamoManager, AlamoManaged


class AlamoModule(torch.nn.Module, AlamoManaged, ABC):
    def __init__(self,
                 torch_module,
                 group: str,
                 from_interface: AlamoInterface = None,
                 to_interface: AlamoInterface = None,):
        super().__init__()
        AlamoManaged.__init__(self)
        self.group = group
        self.torch_module = torch_module
        self.from_interface = from_interface
        self.to_interface = to_interface

    # Borrowed from Flower (https://flower.dev/docs/framework/tutorial-series-get-started-with-flower-pytorch.html)
    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.torch_module.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.torch_module.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.torch_module.load_state_dict(state_dict, strict=True)

    def retrieve_batch(self, batch):
        pass

    def retrieve_gradient(self, grad):
        pass

    def send_gradient(self, grad):
        pass

    def send_batch(self, batch):
        pass

    def start_forward(self):
        pass

    def end_forward(self):
        pass

    def forward(self, batch):
        if not self.alamo_initialised():
            return self.torch_module(batch)
        else:
            self.start_forward()
            if self.alamo_manager.group != self.group:
                # This module should not be computed
                # Actions: all input and outputs will be ignored either way, so it does not matter
                _output = batch
            else:
                # This module should be computed
                # Actions: ensure that output are retrieved and sent during forward pass
                #          ensure that gradients are retrieved and sent during backward pass (through hook)

                class GradientOverwrite(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        return x

                    @staticmethod
                    def backward(ctx, g):
                        return self.retrieve_gradient(g)

                def gradient_hook(grad):
                    self.send_gradient(grad)
                    return grad

                _input = batch if self.from_interface is None \
                    else self.retrieve_batch(batch)  # forward pass: retrieve batch
                _output = self.torch_module(_input)

                if self.from_interface is not None:
                    _output.register_hook(gradient_hook)  # backward pass: send gradient

                if self.to_interface is not None:
                    # forward pass: send output
                    self.send_batch(_output)
                    # backward pass: retrieve gradient
                    gradient_overwrite = GradientOverwrite()
                    _output = gradient_overwrite(_output)
            self.end_forward()
            return _output

