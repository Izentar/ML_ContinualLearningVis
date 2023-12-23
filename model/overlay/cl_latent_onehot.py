from typing import Any, Dict
import torch
from loss_function.chiLoss import OneHot
from utils.cyclic_buffer import CyclicBufferByClass
from utils import pretty_print as pp

from dataclasses import dataclass

from model.overlay.cl_latent import ClLatent

class ClLatentOneHot(ClLatent):
    @dataclass
    class Latent(ClLatent.Latent):
        @dataclass
        class OneHot():
            type: str = None
            scale: float = 1.
            special_class: int = 1

    def __init__(self, *args, **kwargs):
        kwargs.pop('loss_f', None)
        super().__init__(*args, loss_f=torch.nn.MSELoss(), **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=self.cfg.num_classes, dimensions=self.cfg_latent.size, size_per_class=self.cfg_latent_buffer.size_per_class)
        
        onehot_means = self.get_onehots(self.cfg_onehot.type)
        onehot_means_to_print = []
        for k, v in onehot_means.items():
            onehot_means_to_print.insert(k, v)
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Selected onehot type: {self.cfg_onehot.type}\nmeans:\n{torch.stack(onehot_means_to_print)}\n")
        self._loss_f = OneHot(onehot_means, self.cyclic_latent_buffer, loss_f=self._loss_f)

        self.valid_correct = 0
        self.valid_all = 0

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_onehot': ClLatentOneHot.Latent.OneHot,
            #'': CLModelIslandsOneHot.Config,
            'cfg_latent': ClLatentOneHot.Latent,
        })

        b.update({
            'latent.onehot': 'cfg_onehot',
            'latent': 'cfg_latent',
        })
        return a, b

    def get_onehots(self, mytype):
        if(mytype == 'one_cl'):
            d = {}
            for i in range(self.cfg.num_classes):
                d[i] = torch.zeros((self.cfg_latent.size,), dtype=torch.float)
            for k, v in d.items():
                if(k == self.cfg_onehot.special_class):
                    v[self.cfg_latent.size-1] = 1 * self.cfg_onehot.scale
                else:
                    v[0] = 1 * self.cfg_onehot.scale
            return d
        elif(mytype == 'diagonal'):
            d = {}
            counter = 0
            step = 0
            for i in range(self.cfg.num_classes):
                d[i] = torch.zeros((self.cfg_latent.size,), dtype=torch.float)
                if(counter % self.cfg_latent.size == 0):
                    step += 1
                for s in range(step):
                    d[i][(counter + s) % self.cfg_latent.size] = 1 * self.cfg_onehot.scale
                counter += 1
            return d
        elif(mytype == 'random'):
            classes = {}
            for s in range(self.cfg.num_classes):
                classes[s] = torch.rand(self.cfg_latent.size) * self.cfg_onehot.scale
            return classes
        else:
            raise Exception(f"Bad value, could not find means in OneHot config. Key: {mytype}")

    def call_loss(self, input, target, train, **kwargs):
        return self._loss_f(input, target, train)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def decode(self, target):
        return self._loss_f.decode(target)

    def process_losses_normal(self, x, y, latent, log_label, optimizer_idx, model_out_dict=None):
        loss = self._loss_f(latent, y)
        self.log(f"{log_label}/MSE_loss", loss)

        # log values of a point
        selected_class = 0
        uniq = torch.unique(y)
        cl_batch = None
        for u in uniq:
            if(u.item() == selected_class):
                cl_batch = latent[y == u]
                break
        if(cl_batch is not None):
            cl_batch = torch.mean(cl_batch, dim=0)
            for idx, m in enumerate(cl_batch):
                self.log(f"val_cl0_dim{idx}", m)

        return loss

    def process_losses_dreams(self, x, y, latent, log_label, optimizer_idx, model_out_dict=None):
        return self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label=log_label,
            optimizer_idx=optimizer_idx
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        y_model, _ = self.get_model_out_data(model_out)
        val_loss = self._loss_f(y_model, y, train=False)
        self.log("val_last_step_loss", val_loss, on_epoch=True)

        classified_to_class = self._loss_f.classify(y_model)
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(classified_to_class, y)
        self.log("valid_acc", valid_acc)

    def valid_to_class(self, classified_to_class, y):
        uniq = torch.unique(classified_to_class)
        for i in uniq:
            selected = (y == i)
            correct_sum = torch.logical_and((classified_to_class == i), selected).sum().item()
            target_sum_total = selected.sum().item()
            if(target_sum_total != 0):
                pass
                pp.sprint(f'{pp.COLOR.NORMAL_3}Cl {i.item()}: {correct_sum / target_sum_total}')
            self.valid_correct += correct_sum
            self.valid_all += target_sum_total
        pp.sprint(f"{pp.COLOR.NORMAL_3}", self.valid_correct, self.valid_all, self.valid_correct/self.valid_all)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        y_model, _ = self.get_model_out_data(model_out)
        test_loss = self._loss_f(y_model, y, train=False)
        self.log("test_loss", test_loss, on_step=True)

        classified_to_class = self._loss_f.classify(y_model)
        self.test_acc(classified_to_class, y)
        #self.valid_to_class(classified_to_class, y)
        self.log("test_step_acc", self.test_acc)

    def get_obj_str_type(self) -> str:
        return 'ClLatentOneHot_' + super().get_obj_str_type()
