import torch
import math
import numpy as np
import wandb
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import abstractclassmethod

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

class DummyLoss(torch.nn.Module):
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

class ChiLossBase(torch.nn.Module):
    def __init__(self, cyclic_latent_buffer):
        super().__init__()
        self._cloud_data = cyclic_latent_buffer
        self.to_log = {}
        self._train = True

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

    def _get_means_with_target(self, example:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            example - tensor example. How the new tensor means and target should be created.
        """
        means = []
        target = []
        cloud_centers = self._cloud_data.mean()
        for key, val in cloud_centers.items():
            means.append(val.to(example.device))
            target.append(torch.tensor(key, dtype=torch.int8, device=example.device))
        assert len(means) != 0
        #if(len(means) == 0):
        #    return torch.zeros_like(example, dtype=torch.float32), torch.zeros((len(example[0]),), dtype=torch.int8, device=example.device)
        return torch.stack(means, 0), torch.stack(target, 0)
            
    def classify(self, input:torch.Tensor) -> torch.Tensor:
        """
            Return only a vector of classes that will be compared to real batch target
        """ 
        input = input.detach()
        means, target = self._get_means_with_target(example=input)
        target = target.repeat((len(input), 1))
        matrix = torch.cdist(input, means)
        idxs = torch.argmin(matrix, dim=1, keepdim=True)
        classes = torch.gather(target, 1, idxs).squeeze_()
        assert len(classes.shape) != 0
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
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, sigma=0.2, rho=1., eps=1e-5, start_mean_buff_at=500, log_np_loss=True):
        ChiLossBase.__init__(self, cyclic_latent_buffer=cyclic_latent_buffer)
        ChiLossFunctional.__init__(self)
        
        self.sigma = sigma
        self.rho = rho
        self.loss_means_from_buff = loss_means_from_buff
        self.start_mean_buff_at = start_mean_buff_at
        self.log_np_loss = self._log_np_loss_f if log_np_loss else lambda x, y: None

        #if(sigma >= rho):
        #    raise Exception(f"Sigma cannot be bigger or equal than rho - sigma: {sigma}; rho: {tho}")

        # if the loss is nan, change to bigger value
        self.eps = eps  # to not have log(0) problem  
        self.call_idx = 0
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.rand_direction = {}

        self.to_log['rho_sigma'] = (self.rho/self.sigma)**2

    def _log_np_loss_f(self, positive_loss, negative_loss):
            self.to_log['positive_loss'] = positive_loss.detach()
            self.to_log['negative_loss'] = negative_loss.detach()

    def __str__(self) -> str:
        return 'CHI_LOSS'

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

        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()

        means = self._calc_mean_dist(input, target) #/ 2# np.sqrt(2)

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        z_negative = torch.cdist(means, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        negative_loss = (first_part_negative + second_part_negative) * negative_mask     

        positive_loss = self.remove_diagonal(matrix=positive_loss, batch_size=batch_size).sum()
        negative_loss = self.remove_diagonal(matrix=negative_loss, batch_size=batch_size).sum()

        negative_loss = (negative_loss * (self.rho/self.sigma)**2)

        self.log_np_loss(positive_loss=positive_loss, negative_loss=negative_loss)
        loss = positive_loss + negative_loss
        return loss
        
class ChiLossInputFromMeans(ChiLoss):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, sigma=0.2, rho=1, eps=0.00001, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer, loss_means_from_buff, sigma, rho, eps, start_mean_buff_at)

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
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, sigma=0.2, rho=1, eps=0.00001, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer, loss_means_from_buff, sigma, rho, eps, start_mean_buff_at)
    
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
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, sigma=0.2, rho=1, eps=0.00001, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer, loss_means_from_buff, sigma, rho, eps, start_mean_buff_at)

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
        ChiLossBase.__init__(self, cyclic_latent_buffer=cyclic_latent_buffer)
        ChiLossFunctional.__init__(self)

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
