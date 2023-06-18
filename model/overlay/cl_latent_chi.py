from typing import Any, Dict
import torch
from loss_function.chiLoss import ChiLoss, l2_latent_norm
from utils.cyclic_buffer import CyclicBufferByClass
import wandb
import pandas as pd
from utils import pretty_print as pp

from dataclasses import dataclass

from model.overlay.cl_latent import ClLatent
from model.overlay.cl_model import ClModel

class ClLatentChi(ClLatent):
    @dataclass
    class Config(ClModel.Config):
        norm_lambda: float = 0.

    @dataclass
    class Loss():
        @dataclass
        class Chi():
            sigma: float = 0.2
            rho: float = 0.2

    def __init__(
            self, 
            *args, 
            buff_on_same_device=False,  
            **kwargs
        ):
        kwargs.pop('loss_f', None)
        super().__init__(*args, **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=self.cfg.num_classes, dimensions=self.cfg_latent.size, size_per_class=self.cfg_latent_buffer.size_per_class)
        loss_f = ChiLoss(sigma=self.cfg_loss_chi.sigma, rho=self.cfg_loss_chi.rho, cyclic_latent_buffer=self.cyclic_latent_buffer, loss_means_from_buff=False)
        self._setup_loss_f(loss_f)


        self.norm = l2_latent_norm
        self.buff_on_same_device = buff_on_same_device

        self._means_once = False

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg': ClLatentChi.Config,
            'cfg_latent': ClLatentChi.Latent,
            'cfg_loss_chi': ClLatentChi.Loss.Chi,
        })
        b.update({
            'latent': 'cfg_latent',
            'loss.chi': 'cfg_loss_chi',
        })
        return a, b

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self._loss_f(latent, y)

        if(self.cfg.norm_lambda != 0.):
            norm = self.norm(latent, self.cfg.norm_lambda)
            self.log(f"{log_label}/norm", norm)
            loss += norm
        self.log(f"{log_label}/island", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        val_loss = self._loss_f(latent, y, train=False)
        self.log("val_last_step_loss", val_loss, on_epoch=True)
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(self._loss_f.classify(latent), y)
        self.log("valid_acc", valid_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss = self._loss_f(latent, y, train=False)
        self.log("test_loss", test_loss, on_step=True)

        classified_to_class = self._loss_f.classify(latent)
        self.test_acc(classified_to_class, y)
        self.log("test_acc", self.test_acc)

        self._test_step_log_data()

    def _test_step_log_data(self):
        # log additional data only once
        if(not self._means_once):
            new_means = {str(k): v for k, v in self.loss_f.cloud_data.mean().items()}
            distance_key, dist_val = self._calc_distance_to_each_other(new_means)
            table_distance = wandb.Table(columns=distance_key, data=[dist_val])
            wandb.log({'chiloss/distance_to_means': table_distance})

            means_val = self._parse_bar_plot(new_means)
            for k, v in means_val.items():
                table_means = wandb.Table(data=v, columns=['dimension', 'value'])
                wandb.log({f'chiloss/means_cl-{k}': wandb.plot.bar(table_means, label='dimension', value='value', title=f"Class-{k}")})

            new_std = {str(k): v for k, v in self.loss_f.cloud_data.std().items()}
            std_key_class, std_val = self._parse_std(new_std)
            table_std = wandb.Table(columns=std_key_class, data=[std_val])
            wandb.log({'chiloss/std_for_classes': table_std})
            
            self._means_once = True

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().load_checkpoint(checkpoint)
        if(loss_f := checkpoint.get('loss_f')):
            self._loss_f = loss_f
        else:
            pp.sprint(f"{pp.COLOR.WARNING}WARNING: model loss function not loaded. Using default constructor.")
        if(buffer := checkpoint.get('buffer')):
            self.cyclic_latent_buffer = buffer
        else:
            pp.sprint(f"{pp.COLOR.WARNING}WARNING: model latent buffer not loaded. Using default constructor.")
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint['loss_f'] = self._loss_f
        checkpoint['buffer'] = self.cyclic_latent_buffer

    def _parse_std(self, data:dict):
        key = []
        value = []
        for k, v in data.items():
            for k2, v2 in enumerate(v):
                key.append(f"cl={k}::dim={k2}")
                value.append(v2)
        return key, value

    def _parse_bar_plot(self, data:dict):
        vals = {}
        for k, v in data.items():
                vals[k] = []
                for k2, v2 in enumerate(v):
                    vals[k].append([k2, v2])
        return vals

    def _calc_distance_to_each_other(self, means):
        pdist = torch.nn.PairwiseDistance(p=2)
        keys = []
        vals = []
        for k, v in means.items():
            for k2, v2 in means.items():
                keys.append(f"{k}:{k2}")
                vals.append(pdist(v, v2))
        return keys, vals

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if(self.cyclic_latent_buffer is not None and self.buff_on_same_device):
            self.cyclic_latent_buffer.to(self.device)
        return super().training_step(batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx)

    def get_obj_str_type(self) -> str:
        return 'CLModelWithIslands_' + super().get_obj_str_type()
        