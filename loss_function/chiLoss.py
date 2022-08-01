import torch

# input - class / pos
#        pos1, pos2, pos3
# batch1
# batch2
# batch3

Tensor = torch.Tensor


class ChiLoss:
    def __init__(self, sigma=0.2, eps=1e-5):
        self.sigma = sigma

        # if the loss is nan, change to bigger value
        self.eps = eps  # to not have log(0) problem
        self.z_div = (2 * self.sigma**2)

    def __call__(self, input, target):
        k = input.size(dim=1)
        z = torch.pow(torch.cdist(input, input, p=2), 2.).div(self.z_div)
        dims = (k / 2 - 1) 
        dims += (dims == 0) + (dims >= 0) - 1 # if k=2 -> dims=1; k=4 -> dims=1

        first_part = torch.mul(torch.log_(torch.add(torch.div(z, k), self.eps)), -dims)
        second_part = z.div(2 * k)

        target_stacked = target.repeat((len(target), 1))
        z_class = torch.mul(torch.eq(target_stacked, target_stacked.T).float(), 2).add_(-1)

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

        return torch.matmul(first_part.add_(second_part), (z_class)).sum()


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


    

    
