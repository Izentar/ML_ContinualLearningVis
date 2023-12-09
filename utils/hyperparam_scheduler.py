


class HyperparameterSchedulerFloat(float):
    def __new__(cls, param, gamma, milestones):
        return super().__new__(cls, param)

    def __init__(self, param, gamma, milestones) -> None:
        self.param = param
        self.unit = param
        self.gamma = gamma
        self.milestones = milestones
        self.counter = 0

    def step(self) -> None:
        self.counter += 1
        if(self.milestones is None):
            return
        if(self.counter in self.milestones):
            self.unit = self.gamma * self.unit
