from abc import ABC, abstractmethod
from typing import *
from uuid import UUID, uuid5

import torch

from alamo.network import AlamoNetwork


class ConfigLiteral:
    def __init__(self, config_name, type_constraint=None):
        self.config_name = config_name
        self.type_constraint = type_constraint

    def match_config(self, config_dict: Dict) -> Any:
        if self.config_name in config_dict.keys():
            if self.type_constraint is None:
                return config_dict[self.config_name]
            elif any([isinstance(config_dict[self.config_name], t) for t in self.type_constraint]):
                return config_dict[self.config_name]
            else:
                raise TypeError(f"Configuration {self.config_name}: {type(self.config_name)}"
                                f"is not of type {self.type_constraint}.")
        else:
            raise KeyError(f"Configuration {self.config_name} not found in config dict.")


class AlamoManaged(ABC):
    def __init__(self):
        self._alamo_manager = None

    @property
    def alamo_manager(self):
        return self._alamo_manager

    def alamo_initialised(self) -> bool:
        return self._alamo_manager is not None


class AlamoManager:
    def __init__(self,
                 session_uuid: UUID,
                 group: Union[str, int],
                 role: str,
                 net: AlamoNetwork,
                 net_id,
                 node_uuid: UUID = None, ):
        self.session_uuid = session_uuid
        self.group = group
        self.role = role
        if node_uuid is None:
            self.node_uuid = uuid5(session_uuid, role)
        else:
            self.node_uuid = node_uuid
        self.net = net
        self.net_id = net_id
        self.config_dict = None

    def init(self, module,
             config_dict: Dict = None,
             recursion_limit: int = 10):

        self.config_dict = config_dict

        def _init_helper(obj, recursion_level):
            if recursion_level > recursion_limit:
                return obj

            if isinstance(obj, AlamoManaged):
                obj._alamo_manager = self

            if isinstance(obj, torch.nn.Module) or isinstance(obj, AlamoManaged):
                for child_attr in dir(obj):
                    setattr(child_attr, _init_helper(getattr(obj, child_attr), recursion_level + 1))
            elif isinstance(obj, ConfigLiteral):
                _val = obj.match_config(config_dict)
                return _val

            return obj
        return _init_helper(module, 0)

