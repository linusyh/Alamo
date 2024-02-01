from alamo.manager import AlamoManaged

class AlamoInterface(AlamoManaged):
    def __init__(self):
        self._alamo_manager = None

    def get_alamo_manager(self):
        return self._alamo_manager

    def alamo_initialised(self) -> bool:
        return self._alamo_manager is not None


