from abc import abstractmethod

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

    def reset(self):
        self.value = self.start

class CounterKeys(CounterKeysBase):
    def __init__(self, start=0, keys=None) -> None:
        self.keys = keys if keys is not None else [0]
        self.start = start
        self.values = {}
        for i in self.keys:
            self.values[i] = start

    def _create_exist(self, key):
        if(key not in self.keys):
            self.values[key] = self.start

    def up(self, key):
        self._create_exist(key)
        self.values[key] += 1

    def get(self, key):
        self._create_exist(key)
        return self.values[key]

    def reset(self, key=None):
        if(key is None):
            for i in self.keys:
                self.values = self.start
        else:
            self.values[key] = self.start