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

input_data_tensor_1 = torch.ones((3, 3, 32, 32))
output_data_tensor_1 = torch.tensor([0, 1, 2])
logits_data_tensor_1 = torch.ones((3, 50)) + 0.5

class TestCLModel(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(42)
        num_tasks = 5
        num_classes = num_tasks * 10
        epochs_per_task = 15
        dreams_per_target = 48

        train_with_logits = True
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

        dream_dataset_class = dream_sets.DreamDatasetWithLogits if train_with_logits else dream_sets.DreamDataset
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
            dreams_with_logits=train_with_logits,
            train_normal_robustly=train_normal_robustly,
            train_dreams_robustly=train_dreams_robustly,
        ).to('cpu')

    def test_train_step_normal(self):
        ret = torch.tensor(3.904062271118164).to('cpu')
        x = self.model.training_step_normal(batch=(input_data_tensor_1, output_data_tensor_1))
        tst.assert_close(x, ret)

    def test_training_step_dream(self):
        ret = torch.tensor(0.6797193288803101).to('cpu')
        x = self.model.training_step_dream(batch=(input_data_tensor_1, logits_data_tensor_1, output_data_tensor_1))
        tst.assert_close(x, ret)

class TestCLModelWithReconstruction(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(42)
        num_tasks = 5
        num_classes = num_tasks * 10
        epochs_per_task = 15
        dreams_per_target = 48

        train_with_logits = True
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

        dream_dataset_class = dream_sets.DreamDatasetWithLogits if train_with_logits else dream_sets.DreamDataset
        dataset_class = getDataset(args_dataset)
        val_tasks_split = train_tasks_split = datMan.task_split_classic(num_classes, num_tasks)
        select_dream_tasks_f = datMan.select_task_decremental
        dataset_class_robust = getDatasetList(args_dataset)[1]
        dataset_robust = dataset_class_robust(data_path="./data", num_classes=num_classes)
        model_overlay = getModelType(auxiliary_reconstruction)

        self.model = CLModelWithReconstruction(
            model=SAE_CIFAR(num_classes=num_classes),
            robust_dataset=dataset_robust,
            num_tasks=num_tasks,
            num_classes=num_classes,
            attack_kwargs=attack_kwargs,
            dreams_with_logits=train_with_logits,
            train_normal_robustly=train_normal_robustly,
            train_dreams_robustly=train_dreams_robustly,
        ).to('cpu')

    def test_train_step_normal(self):
        ret = torch.tensor(0.43149739503860474).to('cpu')
        x = self.model.training_step_normal(batch=(input_data_tensor_1, output_data_tensor_1))
        tst.assert_close(x, ret)

    def test_training_step_dream(self):
        ret = torch.tensor(0.6797193288803101).to('cpu')
        x = self.model.training_step_dream(batch=(input_data_tensor_1, logits_data_tensor_1, output_data_tensor_1))
        tst.assert_close(x, ret)