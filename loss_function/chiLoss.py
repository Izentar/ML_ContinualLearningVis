import torch
import math
import wandb

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

class ChiLossBase():
    def __init__(self, cyclic_latent_buffer):
        self.centers_of_clouds = cyclic_latent_buffer

    def __call__(self, input, target):
        self.centers_of_clouds.push_target(input, target)

    def _get_stack_means(self, example):
        """
            example - tensor example. How the new tensor target and other should be created.
        """
        means = []
        target = []
        cloud_centers = self.centers_of_clouds.mean()
        for key, val in cloud_centers.items():
            means.append(val.to(example.device))
            target.append(torch.tensor(key, dtype=torch.int8, device=example.device))
        if(len(means) == 0):
            return torch.zeros_like(example, dtype=torch.float32), torch.zeros((len(example[0]),), dtype=torch.int8, device=example.device)
        return torch.stack(means, 0), torch.stack(target, 0)
            
    def classify(self, input):
        # return only a vector of classes that will be compared to real batch target
        input = input.detach()
        means, target = self._get_stack_means(example=input)
        target = target.repeat((len(input), 1))
        matrix = torch.cdist(input, means)
        idxs = torch.argmin(matrix, dim=1, keepdim=True)
        classes = torch.gather(target, 1, idxs)
        classes = torch.squeeze(classes)
        return classes

class ChiLoss(ChiLossBase):
    def __init__(self, cyclic_latent_buffer, loss_means_from_buff, sigma=0.2, rho=1., eps=1e-5, start_mean_buff_at=500):
        super().__init__(cyclic_latent_buffer=cyclic_latent_buffer)
        
        self.sigma = sigma
        self.rho = rho
        self.loss_means_from_buff = loss_means_from_buff
        self.start_mean_buff_at = start_mean_buff_at
        self.rand_distance_scale = 1000000

        self.to_log = {}

        #if(sigma >= rho):
        #    raise Exception(f"Sigma cannot be bigger or equal than rho - sigma: {sigma}; rho: {tho}")

        # if the loss is nan, change to bigger value
        self.eps = eps  # to not have log(0) problem  
        self.call_idx = 0
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.rand_direction = {}

        self.start_latent_means = {}
        self.forget_after = 10e+8
        self.forget_counter_by_class = {}
        self.forget_call_counter = 0
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def to(self, device):
        self.start_latent_means = self._change_device(device, self.start_latent_means)

    def _change_device(self, device, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                new_dict[k] = v.to(device)
            return new_dict
        if isinstance(obj, list):
            new_list = []
            for v in obj:
                new_list.append(v)
            return new_list
        raise Exception(f"Unknown type: {type(obj)}")
        
    def _calculate_batch_mean(self, input, target):
        '''
            Calculate mean only for given batch.
        '''
        unique, inverse, count = torch.unique(target, return_counts=True, return_inverse=True, dim=0)
        means = {}
        for u in unique:
            current = input[target == u]
            means[u.item()] = torch.mean(current, dim=0)

        return self._fill_target_means(target, means)

    def _fill_target_means(self, target, means):
        mean_batch = []
        for t in target:
            mean_batch.append(means[t.item()])
        return torch.stack(mean_batch)

    def _calc_dist_from_means(self, input, means):
        new_dist = {}
        for cl, mean in means.items():
            selected = input[target == cl]
            new_dist[cl] = self.pdist(selected, mean)
        return new_dist

    def _calc_mean_dist(self, input, target):
        if(self.loss_means_from_buff):
            self.call_idx += 1
            if(self.call_idx > self.start_mean_buff_at):
                #self.cyclic_buffer.push_target(input, target)
                means = self.cyclic_latent_buffer.mean()
                #dist = self._calc_dist_from_means(input, means)

                mean_distance = self._fill_target_means(target, means)
                return mean_distance.to(target.device)
        
        return self._calculate_batch_mean(input, target)

    def _get_random_direction(self, batch, target):
        new_batch = []
        for p, t in zip(batch, target):
            item = t.item()
            if(item not in self.rand_direction):
                self.rand_direction[item] = torch.rand(len(p), requires_grad=False, device=target.device)
            new_batch.append(self.rand_direction[item])
        return torch.stack(new_batch, 0)

    def _set_strength_random_direction(self, random_direction, base, mean_matrix):
        # negative sum -> log
        new = []
        for idx, (b, m) in enumerate(zip(base, mean_matrix)):
            mean_avg_dist = torch.mean(torch.cat((m[:idx], m[idx+1:])))
            weight = torch.log(torch.abs(m[idx] - mean_avg_dist))
            new.append(b + random_direction[idx] * weight)
        return torch.stack(new, 0)

    def _set_strength(self, batch_means, batch_rand, input_distance, target, k):
        new_batch_means = []
        for mean, randm, t in zip(batch_means, batch_rand, target):
            diagonal = torch.diag(torch.ones_like(input_distance[0]))
            target_cl = torch.where(target == t, True, False)
            target_cl = target.repeat((len(target_cl), 1))
            negative_mask = torch.logical_or(target_cl, target_cl.T).float() * torch.logical_not(diagonal).float()

            #target_cl = torch.where(target == t, 1., 0.)
            #target_cl = target.repeat((len(target_cl), 1))
            #negative_mask = (target_cl != target_cl.T).float()

            self.input_sum = torch.abs(input_distance * negative_mask).sum()

            distance_weight = torch.abs(input_distance * negative_mask).sum() / self.rand_distance_scale
            distance_weight = (k / 2 - 1) * (1 / (distance_weight / k + self.eps) )
            self.distance_weight = distance_weight
            new_batch_means.append(mean + randm * distance_weight)

        return torch.stack(new_batch_means, 0)

    def _distance_diff(self, input, target):
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        #unique, count = torch.unique(target, return_counts=True)
        #main_cl = unique[torch.argmax(count)]

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()

        # must be order (means, input) to iterate over each means.
        means = self._calc_mean_dist(input, target)
        input_rand = self._get_random_direction(input, target)
        means_rand = self._get_random_direction(means, target)
        mean_matrix = torch.cdist(means, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)
        input = self._set_strength_random_direction(input_rand, input, mean_matrix)
        means = self._set_strength_random_direction(input_rand, means_rand, mean_matrix)

        z_distance_diff = torch.cdist(means, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    
        loss_buff = []
        for i, t in zip(z_distance_diff, target):
            tmp_matrix_2 = []
            for j in z_distance_diff:
                tmp_matrix_2.append(i - j)
            tmp_matrix_2 = torch.stack(tmp_matrix_2, 0)

            #target_stacked = torch.where(target == t, target, 0.)
            target_stacked = target.repeat((len(target), 1))
            target_mask_positive = (target_stacked == target_stacked.T).float()
            target_mask_positive *= torch.where(target == t, 1., 0.).repeat((len(target), 1))
            target_mask_negative = (target_stacked != target_stacked.T).float()
            target_mask_negative *= torch.where(target != t, 1., 0.).repeat((len(target), 1))
            
            
            #outer_loop_distance_mask = torch.ones_like(tmp_matrix_2)
            #outer_loop_distance_mask[idx_i, : ] = 0.

            tmp_matrix_negative = tmp_matrix_2 * target_mask_negative #* negative_mask# * outer_loop_distance_mask
            negative_minus = torch.where(tmp_matrix_negative < 0., tmp_matrix_negative, 0.) * -1
            negative_plus = torch.where(tmp_matrix_negative > 0., tmp_matrix_negative, 0.) * 0.001
            #negative_minus = -(k / 2 - 1) * torch.log(negative_minus / k + self.eps) + negative_minus / (2 * k)
            #negative_minus *= target_mask_negative

            #tmp_matrix_positive = tmp_matrix_2 * target_mask_positive
            #positive_minus = torch.where(tmp_matrix_positive < 0., tmp_matrix_positive, 0.) * -1
            #positive_plus = torch.where(tmp_matrix_positive > 0., tmp_matrix_positive, 0.) * 1

            #concat = negative_minus + negative_plus #+ positive_minus# + positive_plus
            #concat = negative_minus + negative_plus + positive_minus + positive_plus
            concat = negative_minus + negative_plus 
            concat = concat.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
            loss_buff.append(concat.sum())

        loss = torch.stack(loss_buff, 0).sum()

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)
        
        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        
        loss += positive_loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1).sum()
        return loss

    def _distance_to_means(self, input, target):
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()

        means = self._calc_mean_dist(input, target)
        rand_direction = self._get_random_direction(input, target)

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        means = self._set_strength(means, rand_direction, z_positive, target, k)
        z_negative = torch.cdist(input, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        negative_loss = z_negative * positive_mask

        self.positive_loss = positive_loss
        self.negative_loss = negative_loss
        loss =  positive_loss + negative_loss * (self.rho/self.sigma)**2
        
        # remove diagonal - distance from, to the same class
        loss = loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
        return loss.sum()

    # ------------------------------------------------------------------------------------

    def _count_batch_class_occurrences(self, key) -> None:
        for k, val in self.forget_counter_by_class.items():
            self.forget_counter_by_class[k] += 1

    def _get_batched_mean(self, input, target, train):
        unique = torch.unique(target)
        assert len(unique) == 1
        u_item = unique.item()
        self.forget_call_counter += 1

        mean = torch.mean(input, dim=0)
        if train:
            self.start_latent_means[u_item] = mean.detach()
            self.forget_counter_by_class[u_item] = 0
            self._count_batch_class_occurrences(u_item)

        # start gathering means
        other_means = []
        for key, val in self.start_latent_means.items():
            if(key != u_item):
                other_means.append(val)

        mean = torch.unsqueeze(mean, dim=0)
        # only for the first occurrence of the new class
        if(len(other_means) == 0):
            return mean, torch.zeros_like(input) # size does not matter here

        return mean, torch.stack(other_means, dim=0)

    def _forget_batched_mean(self, train) -> None:
        if(not train):
            return
        to_delete = []
        for key, val in self.start_latent_means.items():
            if(self.forget_counter_by_class[key] > self.forget_after):
                to_delete.append(key)
                self.forget_counter_by_class[key] = 0
        for d in to_delete:
            del self.start_latent_means[d]

    def _one_class_batch(self, input, target, train=True):
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()

        current_mean, means = self._get_batched_mean(input, target, train=train)
        self._forget_batched_mean(train=train)

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        z_negative = torch.cdist(current_mean, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        # remove diagonal - distance from, to the same class
        positive_loss = positive_loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
        
        negative_loss = first_part_negative + second_part_negative
        #negative_loss *= 1000

        self.to_log['positive_loss'] = positive_loss.detach().sum()
        self.to_log['negative_loss'] = negative_loss.detach().sum()
        loss = positive_loss.sum() + (negative_loss * (self.rho/self.sigma)**2).sum()
        
        return loss

    def _simple_chi_loss(self, input, target):
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

        self.to_log['positive_loss'] = positive_loss.detach().sum()
        self.to_log['negative_loss'] = negative_loss.detach().sum()
        loss = positive_loss + negative_loss
        
        # remove diagonal - distance from, to the same class
        loss = loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
        return loss.sum()

    def _simple_mean_from_batch_chi_loss(self, input, target):
        k = input.size(dim=1)
        batch_size = input.size(dim=0)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()

        means = self._calc_mean_dist(input, target)

        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 
        z_negative = torch.cdist(means, means, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)    

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        negative_loss = (first_part_negative + second_part_negative) * negative_mask        

        self.to_log['positive_loss'] = positive_loss.detach().sum()
        self.to_log['negative_loss'] = negative_loss.detach().sum()
        loss =  positive_loss + negative_loss * (self.rho/self.sigma)**2 
        
        # remove diagonal - distance from, to the same class
        loss = loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
        return loss.sum()

    def _simple_mean_buffer_chi_loss(self, input, target):
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

        self.to_log['positive_loss'] = positive_loss.detach().sum()
        self.to_log['negative_loss'] = negative_loss.detach().sum()
        loss =  positive_loss + negative_loss * (self.rho/self.sigma)**2 
        
        # remove diagonal - distance from, to the same class
        loss = loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
        return loss.sum()

    def __call__(self, input, target, train=True):
        super().__call__(input, target)
        #return self._distance_diff(input, target)
        #return self._distance_to_means(input, target)
        #return self._one_class_batch(input, target, train=train)
        #return self._simple_chi_loss(input, target)
        return self._simple_mean_from_batch_chi_loss(input, target)
        #return self._simple_mean_buffer_chi_loss(input, target)


class ChiLossOneHot(ChiLossBase):
    def __init__(self, one_hot_means:dict, cyclic_latent_buffer, sigma=0.2, rho=1., eps=1e-5, only_one_hot=False):
        super().__init__(cyclic_latent_buffer=cyclic_latent_buffer)
        self.sigma = sigma
        self.rho = rho
        self.one_hot_means = one_hot_means
        self.eps = eps
        self.only_one_hot = only_one_hot

        self.to_log = {}

        if (one_hot_means is None or len(one_hot_means) == 0):
            raise Exception('Empty dictionary.')

        self.pdist = torch.nn.PairwiseDistance(p=2)

    def to(self, device):
        return

    def _create_one_hot_batch(self, target):
        batch = []
        for t in target:
            batch.append(self.one_hot_means[t.item()].detach().to(t.device))
        return torch.stack(batch, 0)

    def _get_means_and_onehot(self, input, target):
        means = []
        one_hot = []
        unique = torch.unique(target)
        for u in unique:
            means.append(torch.mean(input[target == u], dim=0))
            one_hot.append(self.one_hot_means[u.item()].to(u.device))
        return torch.stack(means, 0), torch.stack(one_hot, 0)

    def __call__(self, input, target, train=True):
        super().__call__(input, target)

        if(self.only_one_hot):
            one_hot_batch = self._create_one_hot_batch(target)
            loss = self.pdist(one_hot_batch, input) ** 2
            
            loss = loss.sum()
        else:
            k = input.size(dim=1)
            batch_size = input.size(dim=0)

            target_stacked = target.repeat((len(target), 1))
            positive_mask = (target_stacked == target_stacked.T).float()

            means, one_hot = self._get_means_and_onehot(input, target)

            z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2) 

            first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
            second_part_positive = z_positive / (2 * k)

            positive_loss = (first_part_positive + second_part_positive) * positive_mask
            # remove diagonal - distance from, to the same class
            positive_loss = positive_loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
            
            loss = positive_loss.sum()

            distance = self.pdist(means, one_hot) ** 2
            one_hot_loss = distance.sum()
            self.to_log['one-hot-loss-dist'] = one_hot_loss
            one_hot_loss = one_hot_loss * 5000
            self.distance = distance
            self.one_hot_loss = one_hot_loss
            loss += one_hot_loss
        
        return loss