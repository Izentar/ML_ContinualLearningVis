import unittest
from loss_function.chiLoss import ChiLoss
from torch import testing as tst
import torch
from model.overlay import CLModel, CLModelWithReconstruction
from model.SAE import SAE_CIFAR
from dataset import dream_sets
from multidimensional_dreams import getDataset, getDatasetList, getModelType
from utils import data_manipulation as datMan
import pytorch_lightning as pl
from stats.point_plot import PointPlot, Statistics
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class TestCLModel(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(42)
        num_tasks = 5
        num_classes = num_tasks * 10
        epochs_per_task = 15
        dreams_per_target = 48

        dreams_with_logits = True
        train_normal_robustly = False
        train_dreams_robustly = False
        auxiliary_reconstruction = True
        args_dataset = 'c10'

        attack_kwargs = attack_kwargs = {
            "constraint": "2",
            "eps": 0.5,
            "step_size": 1.5,
            "iterations": 10,
            "random_start": 0,
            "custom_loss": None,
            "random_restarts": 0,
            "use_best": True,
        }

        dream_dataset_class = dream_sets.DreamDatasetWithLogits if dreams_with_logits else dream_sets.DreamDataset
        dataset_class = getDataset(args_dataset)
        val_tasks_split = train_tasks_split = datMan.task_split_classic(num_classes, num_tasks)
        select_dream_tasks_f = datMan.select_task_decremental
        dataset_class_robust = getDatasetList(args_dataset)[1]
        dataset_robust = dataset_class_robust(data_path="./data", num_classes=num_classes)
        model_overlay = getModelType(auxiliary_reconstruction)

        self.model = CLModel(
            model=SAE_CIFAR(num_classes=num_classes),
            robust_dataset=dataset_robust,
            num_tasks=num_tasks,
            num_classes=num_classes,
            attack_kwargs=attack_kwargs,
            dreams_with_logits=dreams_with_logits,
            train_normal_robustly=train_normal_robustly,
            train_dreams_robustly=train_dreams_robustly,
        ).to('cpu')

        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CIFAR10(root="./data", train=False, transform=transform)
        self.dataloader = DataLoader(dataset, 
            batch_size=64, 
            num_workers=4, 
            pin_memory=False,
        )


    def test_plot_singular(self):
        x1 = torch.tensor([
            [0, 1, 2 ,3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        x2 = torch.tensor([
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ])
        plotter = PointPlot()
        plotter.plot([(x1, [1, 3, 5]), (x2, [7, 5, 3])], plot_type='singular', show=True, name=None)

    def test_plot_multi(self):
        x1 = torch.tensor([
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
        ])
        x2 = torch.tensor([
            [12, 13, 14],
            [16, 17, 18],
            [20, 21, 22],
        ])
        plotter = PointPlot()
        plotter.plot([(x1, [1, 3, 5]), (x2, [7, 5, 3])], plot_type='multi', show=True, name=None, symetric=False)

    def test_collector(self):
        stats = Statistics()
        def invoker(model, input):
            _, xe_latent, _, _, _ = model.source_model.forward_encoder(
                input,
                with_latent=True,
                fake_relu=False,
                no_relu=False,
            )
            return xe_latent
            
        buffer = stats.collect(model=self.model, dataloader=self.dataloader, num_of_points=100, to_invoke=invoker)
        plotter = PointPlot()
        
        #plotter.plot([(x1, [1, 3, 5]), (x2, [7, 5, 3])], plot_type='singular', show=True)
        plotter.plot(buffer, plot_type='multi', show=True, name=None)