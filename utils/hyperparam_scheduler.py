from utils import pretty_print as pp
import sys
from typing import overload
from typing_extensions import SupportsIndex
from numbers import Number


class HyperparameterSchedulerFloat(Number):
    """
        Class used to schedule change of hyperparameters. For now it is used for float type hyperparameters.
    """
    def __init__(self, param, gamma:float, milestones: list, name:str) -> None:
        self.param = param
        self.current_float: float = param
        self.gamma = gamma
        self.milestones = milestones
        self.counter = 0
        self.name = name

    def step(self) -> None:
        self.counter += 1
        if(self.milestones is None):
            return
        if(self.counter in self.milestones):
            current_float = self.gamma * self.current_float
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: Scheduler for {self.name} changed value from: {self.current_float} to: {current_float} at step: {self.counter}")
            self.current_float = current_float

    def __str__(self) -> str:
        return f"{self.current_float}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __add__(self, __value: float) -> float:
        return self.current_float + __value
    def __sub__(self, __value: float) -> float:
        return self.current_float - __value
    def __mul__(self, __value: float) -> float:
        return self.current_float * __value
    def __floordiv__(self, __value: float) -> float: 
        return self.current_float // __value
    def __truediv__(self, __value: float) -> float: 
        return self.current_float / __value
    def __mod__(self, __value: float) -> float: 
        return self.current_float % __value
    def __divmod__(self, __value: float) -> tuple[float, float]: 
        return divmod(self.current_float, __value)
    def __pow__(self, __value: int, __mod: None = None) -> float: 
        return pow(self.current_float, __value, __mod)
    def __radd__(self, __value: float) -> float: 
        return __value + self.current_float
    def __rsub__(self, __value: float) -> float: 
        return __value - self.current_float
    def __rmul__(self, __value: float) -> float: 
        return __value * self.current_float
    def __rfloordiv__(self, __value: float) -> float: 
        return __value // self.current_float
    def __rtruediv__(self, __value: float) -> float: 
        return __value / self.current_float
    def __rmod__(self, __value: float) -> float: 
        return __value % self.current_float
    def __rdivmod__(self, __value: float) -> tuple[float, float]: 
        return divmod(__value, self.current_float)
    def __rpow__(self, __value, __mod: None = None) -> float: 
        return pow(__value, self.current_float, __mod)
    def __getnewargs__(self) -> tuple[float]: 
        return self.current_float.__getnewargs__()
    def __trunc__(self) -> int: 
        return self.current_float.__trunc__()
    if sys.version_info >= (3, 9):
        def __ceil__(self) -> int: 
            return self.current_float.__ceil__()
        def __floor__(self) -> int: 
            return self.current_float.__floor__()
        
    @overload
    def __round__(self, __ndigits: None = None) -> int: 
        return self.current_float.__round__(__ndigits)
    @overload
    def __round__(self, __ndigits: SupportsIndex) -> float: 
        return self.current_float.__round__(__ndigits)
    def __eq__(self, __value: object) -> bool: 
        return self.current_float.__eq__(__value)
    def __ne__(self, __value: object) -> bool: 
        return self.current_float.__ne__(__value)
    def __lt__(self, __value: float) -> bool: 
        return self.current_float.__lt__(__value)
    def __le__(self, __value: float) -> bool: 
        return self.current_float.__le__(__value)
    def __gt__(self, __value: float) -> bool: 
        return self.current_float.__gt__(__value)
    def __ge__(self, __value: float) -> bool: 
        return self.current_float.__ge__(__value)
    def __neg__(self) -> float: 
        return self.current_float.__neg__()
    def __pos__(self) -> float: 
        return self.current_float.__pos__()
    def __int__(self) -> int: 
        return self.current_float.__int__()
    def __float__(self) -> float: 
        return self.current_float.__float__()
    def __abs__(self) -> float: 
        return self.current_float.__abs__()
    def __hash__(self) -> int: 
        return self.current_float.__hash__()
    def __bool__(self) -> bool: 
        return self.current_float.__bool__()