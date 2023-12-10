import torch
import math
import numpy as np
import wandb
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import abstractclassmethod
from utils import pretty_print as pp

# input - class / pos
#        pos1, pos2, pos3
# batch1
# batch2
# batch3

Tensor = torch.Tensor


def l2_latent_norm(weights, lambd):
    norm = sum(torch.pow(p, 2.).sum() for p in weights)
    return lambd * norm

def l2_norm(model, lambd):   
    # call python sum. It needs to call __add__
    norm = sum(torch.pow(p, 2.).sum() for p in model.parameters())
    return lambd * norm

def l1_norm(model, lambd):
    norm = sum(torch.sum(torch.abs(p)) for p in model.parameters())
    return norm * lambd

class BaseLoss():
    pass

class DummyLoss(torch.nn.Module, BaseLoss):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.to_log = {}
        
    def __call__(self, input, target, train=True):
        return self.loss(input, target)

    def classify(self, input):
        return input

    def sample(self, selected_class, utype='normal'):
        raise Exception("Not implemented")

    def decode(self, target):
        return target

    def __str__(self) -> str:
        return 'DUMMY_LOSS__' + str(type(self.loss).__name__)

    def to(self, device):
        pass

    @property
    def cloud_data(self):
        raise Exception('Not implemented')

class ChiLossBase(torch.nn.Module, BaseLoss):
    def __init__(self, cyclic_latent_buffer):
        super().__init__()
        
        pp.sprint(f"{pp.COLOR.NORMAL_2}CHI-LOSS: Used buffer: {cyclic_latent_buffer}")
        self._cloud_data = cyclic_latent_buffer
        self.to_log = {}
        self._train = True

        #if(self._cloud_data.dimensions <= 2):
        #    raise Exception(f"Current Chiloss implementation requires latent size greater than 2. Currently {self._cloud_data.dimensions}")

    @property
    def cloud_data(self):
        return self._cloud_data

    def __call__(self, input:torch.Tensor, target:torch.Tensor, train:bool=True):
        assert not torch.any(torch.isnan(input)), f"Input is NaN\n{input}"
        if self._train and train:
            self._cloud_data.push_target(input, target)

    def __str__(self) -> str:
        return 'CHI_LOSS_BASE'

    def train(self, flag:bool) -> None:
        self._train = flag

    def _get_means_with_target(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Returns all means and its target tensor converted to given device.
        """
        means = []
        target = []
        cloud_centers = self._cloud_data.mean()
        for key, val in cloud_centers.items():
            means.append(val.to(device))
            target.append(torch.tensor(key, dtype=torch.int8, device=device))
        assert len(means) != 0
        #if(len(means) == 0):
        #    return torch.zeros_like(example, dtype=torch.float32), torch.zeros((len(example[0]),), dtype=torch.int8, device=example.device)
        return torch.stack(means, 0), torch.stack(target, 0)
            
    def classify(self, input:torch.Tensor) -> torch.Tensor:
        """
            Return only a vector of classes that will be compared to real batch target
        """ 
        input = input.detach()
        means, target = self._get_means_with_target(device=input.device)
        # creates something like 
        # 0 1 2 3
        # 0 1 2 3
        # matrix B x latent_size
        target = target.repeat((len(input), 1))
        # distance between two matrixes
        matrix = torch.cdist(input, means)
        # select indices of the input that have minimal distance 
        idxs = torch.argmin(matrix, dim=1, keepdim=True)
        classes = torch.gather(target, 1, idxs).squeeze_()
        #assert len(classes.shape) != 0
        #if (len(classes.shape) == 0):
        #    classes = classes.unsqueeze(0)
        return classes

    def sample(self, selected_class, utype='multivariate') -> torch.Tensor:
        """
            Return a value drawn from latent cloud from some distribution like:
                - multivariate normal distribution, characterized by loc(means) and covariance matrix.
        """
        assert (utype == 'multivariate'), f"Not implemented {utype}"
        cloud_mean = self._cloud_data.mean_target(selected_class)
        cloud_cov = self._cloud_data.cov_target(selected_class)
        mnormal = MultivariateNormal(loc=cloud_mean, covariance_matrix=cloud_cov)
        return mnormal.sample()

    def decode(self, target):
        return target

class ChiLossFunctional(torch.nn.Module):
    """
        Class for independent functions.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def remove_diagonal(self, matrix:torch.Tensor, batch_size:int) -> torch.Tensor:
        # remove diagonal - distance from, to the same class
        return matrix.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)

    def _calculate_batch_mean(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        '''
            Calculate mean only for given batch.
        '''
        unique, inverse, count = torch.unique(target, return_counts=True, return_inverse=True, dim=0)
        means = {}
        for u in unique:
            current = input[target == u]
            means[u.item()] = torch.mean(current, dim=0)

        return self._swap_target_to_means(target, means)

    def _swap_target_to_means(self, target:torch.Tensor, means:dict) -> torch.Tensor:
        mean_batch = []
        for t in target:
            mean_batch.append(means[t.item()])
        return torch.stack(mean_batch)
    
    def _change_device_dict(self, device:str, obj:dict) -> dict:
        new_dict = {}
        for k, v in obj.items():
            new_dict[k] = v.to(device)
        return new_dict

    def _change_device_list(self, device:str, obj:list) -> list:
        new_list = []
        for v in obj:
            new_list.append(v.to(device))
        return new_list
    
    def _distance_from_target_to_means(self, input:torch.Tensor, target:torch.Tensor, means:dict) -> dict:
        new_dist = {}
        for cl, mean in means.items():
            selected = input[target == cl]
            new_dist[cl] = self.pdist(selected, mean)
        return new_dist

class ChiLoss(ChiLossBase, ChiLossFunctional):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, classes, latent_size, shift_min_distance, shift_std_of_mean, 
                 ratio, scale,  eps=1e-5, 
                 start_mean_buff_at=500, log_np_loss=True):
        super(ChiLoss, self).__init__(cyclic_latent_buffer=cyclic_latent_buffer)
        
        self.scale = scale
        self.ratio = ratio
        self.loss_means_from_buff = loss_means_from_buff
        self.start_mean_buff_at = start_mean_buff_at
        self.log_np_loss = self._log_np_loss_f if log_np_loss else lambda x, y: None

        # can be translated to:
        #(2 * sigma**2) <=> (2 * scale**2)
        #(2 * rho**2) <=> (2 * ratio*scale**2)
        #(rho/sigma)**2) <=> (ratio**2)
        
        self.eps = eps  # to not have log(0) problem  
        self.call_idx = 0
        self.pdist = torch.nn.PairwiseDistance(p=2)

        self.classes = classes
        self.latent_size = latent_size
        self._init_mean_shift(shift_min_distance, shift_std_of_mean)

        self.to_log['scale'] = self.scale
        self.to_log['ratio'] = self.ratio

    def _is_distance_minimal(self, selected_points: list[torch.Tensor], x: torch.Tensor, shift_min_distance):
        for p in selected_points:
            if(torch.linalg.norm(p - x) <= shift_min_distance):
                return False
        return True

    def _init_mean_shift(self, shift_min_distance, shift_std_of_mean):
        """
            Generate mean shifts that are apart enough from each other to not have interference between point clouds.
        """
        pp.sprint(f'{pp.COLOR.NORMAL}INFO: Generating means shifts')
        if(shift_min_distance >= 0.9 * shift_std_of_mean):
            raise Exception(f'Bad value: shift_min_distance {shift_min_distance} and shift_std_of_mean {shift_std_of_mean} ' +
                            'must be in relation shift_min_distance < 0.9 * shift_std_of_mean')
        #self.mean_shift = torch.ones((self.classes, self.latent_size), requires_grad=False)
        selected_points = []
        std_tensor = torch.ones((self.latent_size, )) * shift_std_of_mean
        counter = 0
        while(True):
            p = torch.normal(torch.zeros(self.latent_size), std=std_tensor)
            if(self._is_distance_minimal(selected_points, p, shift_min_distance)):
                selected_points.append(p)
                counter += 1 
                if(counter == self.classes):
                    break
        torch.stack(selected_points)
        self.mean_shift = torch.stack(selected_points).detach()
        pp.sprint(f'{pp.COLOR.NORMAL}INFO: Generated:\n{self.mean_shift}')

        #self.mean_shift.requires_grad_(True)
        #self.mean_shift = torch.nn.parameter.Parameter(self.mean_shift, requires_grad=True)

    def _add_mean_shift(self, input: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        tmp_matrix = [None] * input.size(0)

        for idx, t in enumerate(target):
            tmp_matrix[idx] = self.mean_shift[t]
            
        # cannot add average over batch because it will change position of the points in unpredictable way
        tmp_matrix = torch.stack(tmp_matrix)
        return torch.add(input, tmp_matrix)

    def _log_np_loss_f(self, positive_loss, negative_loss):
            self.to_log['positive_loss'] = positive_loss.detach()
            self.to_log['negative_loss'] = negative_loss.detach()

    def __str__(self) -> str:
        return 'CHI_LOSS'
    
    def _pop_from_dict(self, d):
        d.pop('sigma', '')
        d.pop('eps', '')
        d.pop('rho', '')
        d.pop('scale', '')
        d.pop('ratio', '')
        d.pop('start_mean_buff_at', '')
    
    def __getstate__(self):
        d = self.__dict__.copy()
        self._pop_from_dict(d)
        return d

    def __setstate__(self, state):
        self._pop_from_dict(state)
        self.__dict__.update(state)

    def _calc_mean_dist(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if(self.loss_means_from_buff):
            self.call_idx += 1
            if(self.call_idx > self.start_mean_buff_at):
                cloud_means:dict = self._cloud_data.mean()
                means:torch.Tensor = self._swap_target_to_means(target, cloud_means)
                return means.to(target.device)
        
        return self._calculate_batch_mean(input, target)

    def __call__(self, input, target, train=True) -> torch.Tensor:
        """
            Distance of input from each other and means form each other
        """
        sigma = self.scale
        rho = self.ratio * self.scale
        super().__call__(input, target, train=train)

        k = input.size(dim=1)
        self.mean_shift = self.mean_shift.to(input.device)
        input_2 = self._add_mean_shift(input=input, target=target)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).fill_diagonal_(0).float()
        negative_mask = (target_stacked != target_stacked.T).fill_diagonal_(0).float()

        means = self._calc_mean_dist(input_2, target) #/ 2# np.sqrt(2)

        z_positive = torch.cdist(input_2, input_2, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (sigma**2) 
        z_negative = torch.cdist(means, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive + self.eps)
        second_part_positive = z_positive / 2

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative + self.eps)
        second_part_negative = z_negative / 2

        positive_loss = ((first_part_positive + second_part_positive) * positive_mask).mean()
        negative_loss = ((first_part_negative + second_part_negative) * negative_mask).mean()

        negative_loss = (negative_loss * (rho/sigma)**2)

        self.log_np_loss(positive_loss=positive_loss, negative_loss=negative_loss)
        loss = positive_loss + negative_loss

        self.to_log['rho_sigma'] = (rho/sigma)**2

        return loss
        
class ChiLossShiftMean(ChiLossBase, ChiLossFunctional):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, classes, latent_size, sigma=0.01, eps=1e-5, l2=0.001,
                 start_mean_buff_at=500):
        super(ChiLossShiftMean, self).__init__(cyclic_latent_buffer=cyclic_latent_buffer)
        
        self.sigma = sigma
        self.loss_means_from_buff = loss_means_from_buff
        self.start_mean_buff_at = start_mean_buff_at
        
        self.eps = eps  # to not have log(0) problem  
        self.call_idx = 0
        self.pdist = torch.nn.PairwiseDistance(p=2)

        self.classes = classes
        self.latent_size = latent_size
        self.l2 = l2
        self._init_mean_shift()

    def _init_mean_shift(self):
        self.mean_shift = torch.ones((self.classes, self.latent_size), requires_grad=False)
        self.mean_shift.normal_(std=10)
        #self.mean_shift.requires_grad_(True)
        #self.mean_shift = torch.nn.parameter.Parameter(self.mean_shift, requires_grad=True)

    def _add_mean_shift(self, input: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        tmp_matrix = [None] * input.size(0)

        for idx, t in enumerate(target):
            tmp_matrix[idx] = self.mean_shift[t]
            
        # cannot add average over batch because it will change position of the points in unpredictable way
        tmp_matrix = torch.stack(tmp_matrix)
        return torch.add(input, tmp_matrix), tmp_matrix
        
    def __str__(self) -> str:
        return 'CHI_LOSS_V2'
    
    def _pop_from_dict(self, d):
        d.pop('sigma', '')
        d.pop('eps', '')
        d.pop('l2', '')
        d.pop('start_mean_buff_at', '')
    
    def __getstate__(self):
        d = self.__dict__.copy()
        self._pop_from_dict(d)
        return d

    def __setstate__(self, state):
        self._pop_from_dict(state)
        self.__dict__.update(state)

    def _calc_mean_dist(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        if(self.loss_means_from_buff):
            self.call_idx += 1
            if(self.call_idx > self.start_mean_buff_at):
                cloud_means:dict = self._cloud_data.mean()
                means:torch.Tensor = self._swap_target_to_means(target, cloud_means)
                return means.to(target.device)
        
        return self._calculate_batch_mean(input, target)

    def __call__(self, input, target, train=True) -> torch.Tensor:
        """
            Distance of input from each other and means form each other
        """
        super().__call__(input, target, train=train)
        self.mean_shift = self.mean_shift.to(input.device)

        k = input.size(dim=1)

        input_2, weight_matrix = self._add_mean_shift(input=input, target=target)
        distance = torch.cdist(input_2, input_2, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (self.sigma**2)   

        first_part = -(k / 2 - 1) * torch.log(distance + self.eps)
        second_part = distance / 2

        loss_sum = (first_part + second_part).mean() #+ (self.l2 * torch.linalg.norm(weight_matrix)**2)
        return loss_sum
    
    def clear(self, classes=None, latent_size=None):
        self.classes = classes if classes else self.classes
        self.latent_size = latent_size if latent_size else self.latent_size
        self._init_mean_shift()
        if(self.cyclic_latent_buffer):
            self.cyclic_latent_buffer.clear()


class ChiLossInputFromMeans(ChiLoss):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, ratio=2.5, scale=10., eps=0.00001, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer=cyclic_latent_buffer, loss_means_from_buff=loss_means_from_buff, 
                         ratio=ratio, scale=scale, eps=eps, start_mean_buff_at=start_mean_buff_at)

    def __str__(self) -> str:
        return 'CHI_LOSS_INPUT_FROM_MEANS'

    def __call__(self, input:torch.Tensor, target:torch.Tensor, train:bool=True) -> torch.Tensor:
        """
            Distance of input from each other and from input to means. Everything else is the same as in ChiLoss.
        """
        super().__call__(input, target, train=train)
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()

        means = self._calc_mean_dist(input, target)

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        z_negative = torch.cdist(input, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        negative_loss = (first_part_negative + second_part_negative) * negative_mask        

        positive_loss = self.remove_diagonal(matrix=positive_loss, batch_size=batch_size).sum()
        negative_loss = self.remove_diagonal(matrix=negative_loss, batch_size=batch_size).sum()

        self.log_np_loss(positive_loss=positive_loss, negative_loss=negative_loss)
        loss = positive_loss + negative_loss * (self.rho/self.sigma)**2 
        return loss

class ChiLossSimple(ChiLoss):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, ratio=2.5, scale=10., eps=0.00001, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer=cyclic_latent_buffer, loss_means_from_buff=loss_means_from_buff, 
                         ratio=ratio, scale=scale, eps=eps, start_mean_buff_at=start_mean_buff_at)
    
    def __str__(self) -> str:
        return 'CHI_LOSS_SIMPLE'

    def __call__(self, input:torch.Tensor, target:torch.Tensor, train:bool=True):
        """
            Distance only between inputs.
        """
        super().__call__(input, target, train=train)
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()

        z = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2
        
        z_positive = z * positive_mask / (2 * self.sigma**2) 
        z_negative = z * negative_mask / (2 * self.rho**2) 

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        positive_loss = first_part_positive + second_part_positive
        negative_loss = first_part_negative + second_part_negative       

        positive_loss = self.remove_diagonal(matrix=positive_loss, batch_size=batch_size).sum()
        negative_loss = self.remove_diagonal(matrix=negative_loss, batch_size=batch_size).sum()

        self.log_np_loss(positive_loss=positive_loss, negative_loss=negative_loss)
        loss = positive_loss + negative_loss
        return loss

class ChiLossBatched(ChiLoss):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, ratio=2.5, scale=10., eps=0.00001, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer=cyclic_latent_buffer, loss_means_from_buff=loss_means_from_buff, 
                         ratio=ratio, scale=scale, eps=eps, start_mean_buff_at=start_mean_buff_at)

        self.last_means_from_batch = {}
        self.forget_after = 10e+8
        self.forget_counter_by_class = {}
        self.forget_call_counter = 0

    def __str__(self) -> str:
        return 'CHI_LOSS_BATCHED'

    def _increment_forget_counter(self, u_target) -> None:
        for k in u_target.items():
            self.forget_counter_by_class[k] += 1

    def to(self, device:str) -> None:
        self.last_means_from_batch = self._change_device_dict(device, self.last_means_from_batch)

    def _get_means_from_batch(self, input:torch.Tensor, target:torch.Tensor, train:bool):
        unique = torch.unique(target)
        assert len(unique) > 1
        u_item:list = unique.item()
        self.forget_call_counter += 1

        mean = torch.mean(input, dim=0)
        if train:
            self.last_means_from_batch[u_item] = mean.detach()
            self.forget_counter_by_class[u_item] = 0
            self._increment_forget_counter(u_item)

        # start gathering means
        complement_means = []
        for key, val in self.last_means_from_batch.items():
            if(key not in u_item):
                complement_means.append(val)

        mean = torch.unsqueeze(mean, dim=0)
        # only for the first occurrence of the new class
        if(len(complement_means) == 0):
            return mean, torch.zeros_like(input) # size does not matter here

        return mean, torch.stack(complement_means, dim=0)

    def _forget_means_from_batch(self, train:bool) -> None:
        if(not train):
            return
        for key in list(self.last_means_from_batch.keys()):
            if(self.forget_counter_by_class[key] > self.forget_after):
                del self.last_means_from_batch[key]
                self.forget_counter_by_class[key] = 0

    def __call__(self, input:torch.Tensor, target:torch.Tensor, train:bool=True):
        """
            Distance of input from each other and distance of current means in batch to means not present in batch.
        """
        super().__call__(input, target, train=train)
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()

        means_from_batch, complement_means = self._get_means_from_batch(input, target, train=train)
        self._forget_means_from_batch(train=train)

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        z_negative = torch.cdist(means_from_batch, complement_means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask        
        negative_loss = first_part_negative + second_part_negative

        positive_loss = self.remove_diagonal(matrix=positive_loss, batch_size=batch_size).sum()
        negative_loss = self.remove_diagonal(matrix=negative_loss, batch_size=batch_size).sum()
        self.log_np_loss(positive_loss=positive_loss, negative_loss=negative_loss)

        return positive_loss + (negative_loss * (self.rho/self.sigma)**2)

class ChiLossOneHotBase(ChiLossBase, ChiLossFunctional):
    def __init__(self, one_hot_means:dict, cyclic_latent_buffer, loss_f=torch.nn.MSELoss()):
        super(ChiLossOneHotBase, self).__init__(cyclic_latent_buffer=cyclic_latent_buffer)

        self.one_hot_means = one_hot_means
        self.loss_f = loss_f

        if(one_hot_means is None or len(one_hot_means) == 0):
            raise Exception('Empty dictionary.')

        if(not isinstance(loss_f, torch.nn.Module)):
            raise Exception("Loss function class not a member of torch.nn.Module.")

class OneHot(ChiLossOneHotBase):
    """
        Compare points from batch to onehots using custom loss function.
        Custom loss function should take input and onehot vector targets like MSELoss.
    """
    def __init__(self, one_hot_means:dict, cyclic_latent_buffer, loss_f=torch.nn.MSELoss()):
        super().__init__(one_hot_means=one_hot_means, cyclic_latent_buffer=cyclic_latent_buffer, loss_f=loss_f)

    def __str__(self):
        return 'ONEHOT'

    def to(self, device):
        return
    
    def decode(self, target:torch.Tensor) -> torch.Tensor:
        return self._create_one_hot_batch(target)
    
    def _create_one_hot_batch(self, target:torch.Tensor) -> torch.Tensor:
        batch = []
        for t in target:
            batch.append(self.one_hot_means[t.item()].detach().to(t.device))
        return torch.stack(batch, 0)
        
    def __call__(self, input:torch.Tensor, target:torch.Tensor, train=True) -> torch.Tensor:
        one_hot_batch = self._create_one_hot_batch(target)
        return self.loss_f(input, one_hot_batch.float())

    @property
    def cloud_data(self):
        raise Exception('Not implemented')

class OneHotPairwise(OneHot):
    """
        Compare points from batch to onehots using pairwise distance
    """
    def __init__(self, one_hot_means:dict, cyclic_latent_buffer):
        super().__init__(one_hot_means=one_hot_means, cyclic_latent_buffer=cyclic_latent_buffer, loss_f=torch.nn.MSELoss())

        self.pdist = torch.nn.PairwiseDistance(p=2)

    def __str__(self):
        return 'ONEHOT_PAIRWISE'

    def to(self, device):
        return

    def __call__(self, input:torch.Tensor, target:torch.Tensor, train=True) -> torch.Tensor:
        one_hot_batch = self._create_one_hot_batch(target)
        loss = self.pdist(one_hot_batch, input) ** 2
        loss = loss.sum()
        return loss

class OneHotIslands(OneHotPairwise):
    """
        Use chiloss positive (cdist) on the points from batch from the same class and 
        compare last X classes points means to corresponding onehots.
    """
    def __init__(self, one_hot_means:dict, cyclic_latent_buffer, sigma=0.2, eps=1e-5, onehot_means_distance_scale=5000):
        super().__init__(one_hot_means=one_hot_means, cyclic_latent_buffer=cyclic_latent_buffer)

        self.sigma = sigma
        self.eps = eps
        self.onehot_means_distance_scale = onehot_means_distance_scale

    def __str__(self):
        return 'ONEHOT_ISLANDS'

    def to(self, device):
        return

    def _get_means_and_onehot(self, input:torch.Tensor, target:torch.Tensor):
        means = []
        one_hot = []
        unique = torch.unique(target)
        for u in unique:
            index = (target == u).item()
            means.append(torch.mean(input[index], dim=0))
            one_hot.append(self.one_hot_means[u.item()].to(u.device))
        return torch.stack(means, 0), torch.stack(one_hot, 0)

    def __call__(self, input:torch.Tensor, target:torch.Tensor, train:bool=True) -> torch.Tensor:
        """
            Distance between each input and distance between means calculated for current batch and one hot.
        """
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2)
        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        loss = self.remove_diagonal(matrix=positive_loss, batch_size=batch_size).sum()

        means, one_hot = self._get_means_and_onehot(input, target)

        one_hot_loss = (self.pdist(means, one_hot) ** 2).sum()
        self.to_log['one-hot-loss-dist'] = one_hot_loss

        one_hot_loss = one_hot_loss * self.onehot_means_distance_scale
        loss += one_hot_loss
    
        return loss
