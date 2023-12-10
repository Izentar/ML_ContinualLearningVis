from abc import abstractmethod
from collections.abc import Sequence

class CounterBase():
    @abstractmethod
    def up(self):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class CounterKeysBase():
    @abstractmethod
    def up(self, key):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def reset(self, key=None):
        pass

class Counter(CounterBase):
    def __init__(self, start=0) -> None:
        self.start = start
        self.value = start

    def up(self):
        self.value += 1

    def get(self):
        return self.value

    def pp_get(self):
        self.value += 1
        return self.value

    def get_pp(self):
        tmp = self.value
        self.value += 1
        return tmp

    def reset(self):
        self.value = self.start

class CounterKeys(CounterKeysBase):
    def __init__(self, start=0, keys:Sequence[int]=None) -> None:
        keys = keys if keys is not None else [0]
        self.start = start
        self.values = {}
        for i in keys:
            self.values[i] = start

    def _create_exist(self, key):
        if(key not in self.values):
            self.values[key] = self.start

    def up(self, key):
        self._create_exist(key)
        self.values[key] += 1
        return self.values[key]

    def get(self, key):
        self._create_exist(key)
        return self.values[key]
    
    def __getitem__(self, key):
        return self.get(key)

    def reset(self, key=None):
        if(key is None):
            for i in self.values:
                self.values = self.start
        else:
            self.values[key] = self.start