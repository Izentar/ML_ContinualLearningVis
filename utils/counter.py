

class Counter():
    def __init__(self, start=0) -> None:
        self.start = start
        self.value = start

    def up(self):
        self.value += 1

    def get(self):
        return self.value

    def reset(self):
        self.value = self.start

class CounterKeys():
    def __init__(self, start=0, keys=None) -> None:
        self.keys = keys if keys is not None else [0]
        self.start = start
        self.values = {}
        for i in keys:
            self.values[i] = start

    def up(self, key):
        self.values[key] += 1

    def get(self, key):
        return self.values[key]

    def reset(self, key=None):
        if(key is None):
            for i in self.keys:
                self.values = self.start
        else:
            self.values[key] = self.start