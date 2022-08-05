import torch
import math

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

class ChiLoss:
    def __init__(self, sigma=0.2, rho=0.4, eps=1e-5):
        self.sigma = sigma
        self.rho = rho

        if(sigma >= rho):
            raise Exception(f"Sigma cannot be bigger or equal than rho - sigma: {sigma}; rho: {tho}")

        # if the loss is nan, change to bigger value
        self.eps = eps  # to not have log(0) problem        

    def __call__(self, input, target):
        k = input.size(dim=1)
        batch_size = input.size(dim=0)
        z_positive = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2)
        z_negative = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.rho**2)

        first_part_positive = -(k / 2 - 1) * torch.log(z_positive / k + self.eps)
        second_part_positive = z_positive / (2 * k)

        first_part_negative = -(k / 2 - 1) * torch.log(z_negative / k + self.eps)
        second_part_negative = z_negative / (2 * k)

        target_stacked = target.repeat((len(target), 1))
        positive_mask = (target_stacked == target_stacked.T).float()
        negative_mask = (target_stacked != target_stacked.T).float()
        z_class = positive_mask.float() * 2 - 1 

        positive_loss = (first_part_positive + second_part_positive) * positive_mask
        negative_loss = (first_part_negative + second_part_negative) * negative_mask

        loss = ((positive_loss + negative_loss) * z_class)
        loss = loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)
        return loss.sum()

class TESTChiLoss:
    def __init__(self, sigma=0.2, eps=1e-5):
        self.sigma = sigma

        # if the loss is nan, change to bigger value
        self.eps = eps  # to not have log(0) problem

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        k = input.size(dim=1)
        batch_size = input.size(dim=0)
        dims = (k / 2 - 1) 
        dims += (dims == 0) + (dims >= 0) - 1 # if k=2 -> dims=1; k=4 -> dims=1

        # z should be always positive :)
        z = torch.cdist(input, input, p=2, compute_mode='donot_use_mm_for_euclid_dist') ** 2 / (2 * self.sigma**2)

        target_stacked = target.repeat((len(target), 1))
        positive_mask =  (target_stacked == target_stacked.T).float()
        negative_mask =  (target_stacked != target_stacked.T).float()
        z_class = positive_mask.float() * 2 - 1 

        first_part = -dims * torch.log((z / k) + self.eps)
        second_part = z / (2 * k)
        #third_part = torch.log(((z + 1) / k) + self.eps) + z + 1
        #third_part = 1 / (z + self.eps)
        third_part = second_part
        
        first_part *= positive_mask 
        second_part *= positive_mask 
        third_part *= negative_mask

        #print(torch.abs(z.detach()).sum().item())

        #print('input', input.sum())
        #print('z', z.sum())
        #print('k', k)
        #print('torch.log(z / k)', torch.log(z / k + self.eps))
        #print('torch.log(z / k).sum()', -dims * torch.log(z / k + self.eps))
        #print('first_part', first_part.sum())
        #print('second_part', second_part.sum())
        #print('target_stacked', target_stacked.size())
        #print('target_stacked', target_stacked)
        #print('z_class', z_class.size())
        #print('z_class', z_class)
        #exit()

        loss = (first_part + second_part + third_part)
        loss = loss.flatten()[1:].view(batch_size-1, batch_size+1)[:,:-1].reshape(batch_size, batch_size-1)

        return loss.sum()

class OLDChiLoss():
    def __init__(self, sigma = 0.2, eps=1e-5):
        self.sigma = sigma

        # if the loss is nan, change to bigger value
        self.eps = torch.tensor(eps) # to not have log(0) problem

    def __call__(self, input: Tensor, target: Tensor):
        sigma = torch.tensor(self.sigma, device=target.device) # no grad needed
        dim_classes = torch.squeeze(torch.as_tensor(input[0].size()))
        #batch_size = torch.squeeze(torch.as_tensor(input.size()))
        batch_size = len(input)
        k = dim_classes
        k2 = k.mul(2)
        sig_sqr = torch.square(sigma)

        XXT = torch.matmul(input, torch.transpose(input, 0, 1))
        X2 = torch.square(torch.linalg.norm(input, dim=1))
        X3 = torch.transpose(X2.repeat((batch_size, 1)), 0, 1)
        Z = X3.add(
            torch.transpose(X3, 0, 1)
        ).sub(
            XXT.mul(2)
        )

        D = torch.zeros((len(target), len(target)), dtype=torch.float64)
        loss_sum = torch.tensor(0., requires_grad=True, device=target.device)
        for row_idx, row in enumerate(Z):
            for col_idx, col in enumerate(row):
                z_class = 1 if target[row_idx] == target[col_idx] else -1
                z = Z[row_idx][col_idx]
                #if not torch.is_nonzero(z):
                #    z = self.eps
                calc_main_part = torch.mul(
                    torch.log(z.div(k2).div(sig_sqr) + self.eps),  # TODO sprawdzić czy o taką kolejność dzielenia chodziło
                    k.div(2).sub(1)
                )
                
                calc_last_part = z.mul(0.5).div(k2).div(sig_sqr)
                #print((calc_last_part).sub(calc_main_part))
                #print((calc_last_part).sub(calc_main_part).mul(z_class))
                tmp = (calc_last_part).sub(calc_main_part).mul(z_class)
                #if torch.isinf(tmp) or torch.isnan(tmp):
                #    print(z, k2, sig_sqr)
                #    print(z, sig_sqr, calc_last_part, calc_main_part, z_class)
                D[row_idx][col_idx] = tmp
                
        
        # class 1
        # -(k/2-1)*ln(z/2k/sigma^2) + 0.5*z/2k/sigma^2
        #(calc_last_part).sub(calc_main_part)

        # class -1 = - (class 1)
        loss = torch.sum(D)
        #print(loss)
        return loss
