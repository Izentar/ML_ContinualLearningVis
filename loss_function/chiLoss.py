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

    def __call__(self, input, target):
        k = input.size(dim=1)
        z = torch.cdist(input, input, p=2) ** 2 / (2 * self.sigma**2)

        target_stacked = target.repeat((len(target), 1))
        z_class = (target_stacked != target_stacked.T).long() * 2 - 1

        first_part = -(k / 2 - 1) * torch.log(z / k + self.eps)
        second_part = z / (2 * k)

        return ((first_part + second_part) * z_class).sum()


def chiLoss_target_transform(target: Tensor):
    # NOT USED
    # here we do not need grad history, because target does not come from any NN weight
    main_value = torch.argmax(torch.bincount(targets))

    boolean_idxs = torch.logical_and(targets, main_value.repeat(torch.as_tensor(targets.size()))) # .item
    #main_idxs = torch.squeeze(torch.transpose(torch.nonzero(boolean_idx), 0, 1))
    #part_idxs = torch.squeeze(torch.transpose(torch.nonzero(~boolean_idx), 0, 1))

    return torch.where(boolean_idxs, 1, -1)
