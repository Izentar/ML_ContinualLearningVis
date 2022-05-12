import torch
from abc import abstractmethod
from numpy import random
from itertools import combinations
from torch import nn

Tensor = torch.Tensor


#TODO - maybe use Perlin noise? By default it will use uniform distribution and var will be 1.

class BoxBorder():
    def __init__(self, edge_length):
        """
            edge_length - edge length of the box. It will be centered at the origin of the coordinate system. 
            Return tuple(points var, points)
        """
        self.edge_length = edge_length
        self.half = torch.tensor(edge_length / 2)
        self.nhalf = torch.tensor(- edge_length / 2)

    def exceeds(self, point):
        inside = True
        diff = []
        for p in point:
            default = 0.
            if not (self.half.ge(p) and p.ge(self.nhalf)): # > p > 
                inside = False
                if self.half.le(p): # <=
                    default = p.sum(self.half)
                elif p.le(self.nhalf): # <=
                    default = self.nhalf.sub(p)
            diff.append(default)
        if not inside:
            return False
        return diff

    def generate(self, dim:int, how_many:int) -> list():
        """
            Return list of points sampled from uniform distribution.
        """
        elements = [self.half, self.nhalf]
        box_vertices = combinations(elements, dim)
        
        points=[]
        for i in range(how_many):
            point = []
            for d in range(dim):
                point.append(random.uniform(low=self.nhalf, high=self.half))
            point = torch.tensor(point, requires_grad=True)
            points.append(point)
        points = torch.tensor(points, requires_grad=True)
        return points



#TODO - torch grad is here preserved?
class TargetBuffer():
    def __init__(self, label: Tensor, dim:int, input=None, border_obj=None):
        self.label = label # target class
        self.buffer = None # stores a multiple records of input
        self.var = None
        self.mean = None
        self.dim = dim

        #print("============================")
        #print(f"Creating {border_obj is None}, {input is None}, {dim}, {label}")
        #print("============================")

        if input is not None:
            self.add(input=input, target=label)
        elif border_obj is not None:
            self.mean = torch.unsqueeze(border_obj.generate(dim=dim, how_many=1), dim=0)
            self.var = torch.tensor([1.] * dim, requires_grad=True)

    def can_add(self, target: Tensor):
        if target == self.label:
            return True
        return False

    def add(self, input: Tensor, target: Tensor):
        if target != self.label:
            raise Exception("Wrong target to label")
        if self.buffer is None:
            self.buffer = torch.unsqueeze(input, dim=0) # torch.clone(input), TODO - grad is here preserved?
        else:
            self.buffer = torch.cat(self.buffer, input, 0)  # grad preserved, TODO should it be?
        #print("-------------------------")
        #print(self.label, target, self.buffer)

    def get_var_mean(self):
        """
            Get mean and var over batch.
        """
        return torch.var_mean(self.buffer, 1)

    def update_var_mean(self):  
        if self.buffer is not None:
            self.var, self.mean = self.get_var_mean()
        return self.var, self.mean

    def clear(self):
        self.buffer = None
        self.var = None
        self.mean = None

class AbstractLoss(): 
    # TODO not finished
    @abstractmethod 
    def process_task(self, task):
        pass

    @abstractmethod
    def __call__(
        self,
        input: Tensor,
        target: Tensor
        ):
        pass

    @abstractmethod
    def clear(self):
        pass

class AbstractNormalDistr(AbstractLoss):
    # TODO not finished
    @abstractmethod
    def get_mean_var(self, task):
        pass

class PointScopeLoss(AbstractNormalDistr):
    #TODO this need to be optimized. A lot of memory copying is used here
    # TODO not finished
    def __init__(self, 
        border_obj,
        dim:int,
        var_range= 2.0, # default 2 sigmas
        #closeup_loss_scale=1.0, 
        var_wide_loss_scale=1.5, # should be higher than closeup_loss_scale to move mean around
        var_target_value=1.0,
        main_loss_f=nn.CrossEntropyLoss
    ):
        self.border_obj = border_obj
        self.dim = dim
        self.target_buffer = {}
        self.var_range = var_range
        #self.closeup_loss_scale = closeup_loss_scale
        self.var_wide_loss_scale = var_wide_loss_scale
        self.var_target_value = var_target_value
        self.main_loss_f = main_loss_f

    def get_mean_var(self, task):
        if task not in self.target_buffer:
            self.target_buffer[task] = TargetBuffer(label=task, dim=self.dim, border_obj=self.border_obj)
        buffer = self.target_buffer[task]
        return buffer.mean, buffer.var

    def update(
        self,
        input: Tensor,
        target: Tensor
    ):
        """
            input - predicted values
            target - target values
            Saves the input and target to the container.
        """
        split_input = torch.split(input, 1) # remove additional dimension
        split_target = torch.split(target, 1)
        border_loss = 0.
        print("New loop")
        for inp, targ in zip(split_input, split_target):
            inp = inp.squeeze()
            targ = targ.squeeze()
            if targ not in self.target_buffer:
                print("++++++++++++++++++++++++")
                print(f"Creating {targ}\n{inp}")
                print("++++++++++++++++++++++++")
                self.target_buffer[targ] = TargetBuffer(label=targ, dim=self.dim, input=inp)
            else:
                self.target_buffer[targ].add(input=inp, dim=self.dim, target=targ)

    def __call__(
        self,
        input: Tensor,
        target: Tensor
        ):
        """
            input - predicted values
            target - target values
            Returns the calculated distance loss.
        """
        split_input = torch.split(input, 1)
        split_target = torch.split(target, 1)
        border_loss = torch.tensor(0., requires_grad=True)
        main_loss = torch.tensor(0., requires_grad=True)
        for inp, targ in zip(split_input, split_target):
            exeeds = self.border_obj.exceeds(inp)
            if(exeeds):
                border_loss.add_(torch.sum(self.border_obj.exceeds(inp))) # simple loss as sum

            main_loss.add_(self.__distance_loss(single_input=inp, single_target=targ))

        return main_loss.add_(border_loss)

        #TODO - calculate also covariance that will try to converges to identity matrix. var is only for now.
        #return self.__distance_loss() + border_loss

    def __distance_loss(self, single_input, single_target):
        """
            Calculates the cross_entropy loss over the mean with 
            gaussian variance regularization. The further input is from the mean,
            the more loss is added and if the input is in range of var_range, the 
            variance regularization is not added.
        """
        target_mean = self.target_buffer[single_target].mean
        mean_loss = nn.CrossEntropyLoss(single_input, target_mean)
        distance = torch.unsqueeze(torch.cdist(
            torch.unsqueeze(target_mean, dim=0), # needs 2D
            torch.unsqueeze(mean_loss, dim=0),
            p=2
        ))
        var_border_range = var_target_value * var_range

        if var_border_range > distance:
              return mean_loss

        # add the regularization - if the point is too far away from the mean,
        # then it has more loss.
        return mean_loss.add((var_border_range - distance) * var_wide_loss_scale)

        

    def deprecated__distance_loss(self):
        """

        """
        # every label with every label
        loss_sum = torch.zeros(1)
        for key, val in self.target_buffer.items():
            val.update_var_mean() # needed to have the latest mean and var or when mean is None
            for key2, val2 in self.target_buffer.items():
                #val2.update_var_mean() #TODO temporary solution
                if key != key2:
                    #print("-------------")
                    #print(type(val.mean), key, val.buffer.size(), val.mean)
                    #print(type(val2.mean), key2, val2.buffer.size(), val2.mean)
                    real_mean_distance = torch.cdist(
                        (torch.tensor(val.mean)),
                        (torch.tensor(val2.mean)), p=2.)
                    #real_var_distance = torch.cdist(
                    #    torch.unsqueeze(val.var.mul(self.var_range), 0), 
                    #    torch.unsqueeze(val2.var.mul(self.var_range), 0), p=2.)
                    #real_var_distance = torch.add(
                    #    torch.unsqueeze(val.var, 0), 
                    #    torch.unsqueeze(val2.var, 0)).mul_(self.var_range) #uses math norm 1
                    real_var_distance = torch.squeeze(torch.add(
                        torch.cdist(
                            torch.unsqueeze(torch.tensor(val.var).mul(self.var_range), dim=0),
                            torch.unsqueeze(torch.zeros_like((torch.tensor(val.var))), dim=0),
                            p=2.),
                        torch.cdist(
                            torch.unsqueeze(torch.tensor(val2.var).mul(self.var_range), dim=0),
                            torch.unsqueeze(torch.zeros_like(torch.tensor(val2.var)), dim=0), 
                            p=2.)))
                    distance_loss = self.__are_too_close_loss(real_var_distance, real_mean_distance)
                    loss_sum.add_(distance_loss)
            var_loss = self.__var_too_wide_loss(val.var)
            loss_sum.add_(var_loss)
        
        # average
        tmp_size = len(self.target_buffer)
        return loss_sum.div_(tmp_size * (tmp_size - 1) + tmp_size)
                   
    def __var_too_wide_loss(self, var_distance):
        # var target is a list of ones.
        ones = torch.ones_like(var_distance).mul_(self.var_target_value)
        #TODO should it converge to one or it could be less than one?
        return torch.squeeze(torch.nn.MSELoss(var_distance, ones)).mul_(self.var_wide_loss_scale)

    def __are_too_close_loss(self, var_distance, mean_distance):
        diff = torch.sub(var_distance, mean_distance)
        #if not torch.gt(diff, torch.zeros_like(diff)):
        tmp = diff.item()
        if tmp < 0.0:
            return  torch.squeeze(diff).mul_(self.closeup_loss_scale)
        return torch.zeros(1)

    def clear(self):
        """
            Pass this obj into main loop and invoke it there. 
        """
        self.target_buffer = {}