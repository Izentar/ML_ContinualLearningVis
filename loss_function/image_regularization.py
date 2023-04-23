import torch

class VariationRegularization():
    def __init__(self, scale) -> None:
        self.scale = scale
        self.loss = None

    def __call__(self, images):
        diff1 = images[:,:,:,:-1] - images[:,:,:,1:]
        diff2 = images[:,:,:-1,:] - images[:,:,1:,:]
        diff3 = images[:,:,1:,:-1] - images[:,:,:-1,1:]
        diff4 = images[:,:,:-1,:-1] - images[:,:,1:,1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        self.loss = self.scale * loss_var

    def gather_loss(self, loss):
        return loss + self.loss

class L2Regularization():
    def __init__(self, coefficient) -> None:
        self.coefficient = coefficient
        self.loss = None

    def __call__(self, images):
        self.loss = self.coefficient * torch.norm(images, 2)

    def gather_loss(self, loss):
        return loss + self.loss