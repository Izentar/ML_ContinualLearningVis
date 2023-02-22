import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import gc
from torch.utils.data import Dataset

class DreamDataset(Dataset):
    """
        Class for storing dreams. It implements extend method so it can accumulate them during 
    """
    def __init__(self, transform=None, **kwargs) -> None:
        self.dreams = []
        self.targets = []
        self.transform = transform

    def __len__(self):
        return len(self.dreams)

    def __getitem__(self, idx):
        dream = self.dreams[idx]
        if self.transform:
            dream = self.transform(dream)
        return dream, self.targets[idx]

    def extend(self, new_dreams: torch.Tensor, new_targets: torch.Tensor, model: torch.nn.Module=None):
        if len(new_dreams) != len(new_targets):
            raise Exception(f"Wrong size between new dreams and targets."
            f"\nDreams: {new_dreams.size()}; {len(new_dreams)}\nTargets: {new_targets.size()}; {len(new_targets)}")
        with torch.no_grad():
            for dream, target in zip(new_dreams, new_targets):
                dream = dream.detach().cpu()
                self._generate_additional_data(dream, target, model)
                self.targets.append(target)
                if self.transform:
                    dream = to_pil_image(dream)
                self.dreams.append(dream)

    def clear(self, instant=False):
        if instant:
            del self.dreams
            del self.targets
            gc.collect()
        self.dreams = []
        self.targets = []

    def _generate_additional_data(self, dream, target, model):
        """Dummy method to be overloaded. Implement custom additional dream processing
        here."""

    def empty(self):
        return len(self.dreams) == 0

    def save(self, location:str):
        to_save = {
            'metadata':{
                
            },
        }

class DreamDatasetWithLogits(DreamDataset):
    def __init__(self, enable_robust=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logits = []
        self.enable_robust = enable_robust

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        return batch[0], self.logits[idx], batch[1]

    def clear(self, instant=False):
        if instant:
            del self.logits
        self.logits = []
        super().clear(instant)

    def extend(self, new_dreams, new_targets, model):
        self._regenerate_old_logits(model)
        super().extend(new_dreams, new_targets, model)

    def _generate_additional_data(self, dream, target, model):
        if(self.enable_robust):
            logits = model(dream.unsqueeze(0).to(model.device), make_adv=False,
                        with_image=False)[0]
        else:
            logits = model(dream.unsqueeze(0).to(model.device))[0]
        logits = logits.squeeze().detach().cpu()
        self.logits.append(logits)

    def _regenerate_old_logits(self, model):
        self.logits.clear()
        with torch.no_grad():
            for dream in self.dreams:
                dream = to_tensor(dream)
                self._generate_additional_data(dream, None, model)

