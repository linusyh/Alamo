from abc import ABC, abstractmethod

from alamo.manager import AlamoManaged


class AlamoNetwork(ABC, AlamoManaged):
    @abstractmethod
    def send(self, data, recipient):
        pass

    @abstractmethod
    def receive(self, data, sender):
        pass