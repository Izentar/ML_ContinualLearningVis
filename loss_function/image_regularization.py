import torch

class TotalVariationRegularization():
    """
        Patrz Mahendran, V. 2014. Understanding Deep Image Representations by Inverting Them.
    """
    def __init__(self, scale) -> None:
        self.scale = scale
        self.loss = None

    def __call__(self, images):
        diff1 = images[:,:,:,:-1] - images[:,:,:,1:]
        diff2 = images[:,:,:-1,:] - images[:,:,1:,:]
        #diff3 = images[:,:,1:,:-1] - images[:,:,:-1,1:]
        #diff4 = images[:,:,:-1,:-1] - images[:,:,1:,1:]
        #loss_var = torch.linalg.norm(diff1) + torch.linalg.norm(diff2) + torch.linalg.norm(diff3) + torch.linalg.norm(diff4)
        #self.loss = self.scale * loss_var

        # Use anisotropic version from https://en.wikipedia.org/wiki/Total_variation_denoising
        self.loss = torch.sum(torch.abs(diff1)) + torch.sum(torch.abs(diff2))

    def gather_loss(self, loss):
        return loss + self.loss

class L2Regularization():
    def __init__(self, coefficient) -> None:
        self.coefficient = coefficient
        self.loss = None

    def __call__(self, images):
        self.loss = self.coefficient * torch.linalg.norm(images, 2)**2

    def gather_loss(self, loss):
        return loss + self.loss