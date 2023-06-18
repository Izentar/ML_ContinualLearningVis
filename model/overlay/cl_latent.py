from typing import Any, Dict
from utils import pretty_print as pp

from config.default import datasets, datasets_map
from dataclasses import dataclass

from model.overlay.cl_model import ClModel

class ClLatent(ClModel):
    @dataclass
    class Latent():
        size: int = None

        def post_init_Latent(self, num_classes):
            self.size = self.size if self.size is not None else num_classes

        @dataclass
        class Buffer():
            size_per_class: int = 40

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_latent.post_init_Latent(self.cfg.num_classes)
        

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_latent': ClLatent.Latent,
            'cfg_latent_buffer': ClLatent.Latent.Buffer,
        })

        b.update({
            'latent': 'cfg_latent',
            'latent.buffer': 'cfg_latent_buffer',
        })
        return a, b
