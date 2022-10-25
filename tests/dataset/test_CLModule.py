import unittest
from dataset.CLModule import CLDataModule
from utils import data_manipulation as datMan
import multidimensional_dreams as md
from torchvision import transforms
from torch import testing as tst

from model.SAE import SAE_standalone
from loss_function.chiLoss import ChiLoss
from dataset import dream_sets

from torchvision.datasets import CIFAR10, CIFAR100

class TestCLDataModule(unittest.TestCase):

    def setUp(self):
        self.dreams_per_target = 48
        self.num_classes = 10
        self.num_tasks = 5

        self.model = SAE_standalone(
            num_tasks=self.num_tasks,
            num_classes=self.num_classes,
            loss_f = ChiLoss(sigma=0.2)
        )    

        self.dreams_transforms = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        self.dream_dataset_class = dream_sets.DreamDataset(transform=self.dreams_transforms)

        self.main = CLDataModule(
            train_tasks_split=datMan.task_split_classic(self.num_classes, self.num_tasks),
            dataset_class=md.getDataset("c10"),
            dreams_per_target=self.dreams_per_target,
            val_tasks_split=datMan.task_split_classic(self.num_classes, self.num_tasks),
            select_dream_tasks_f=datMan.select_task_decremental,
            empty_dream_dataset=self.dream_dataset_class,
            fast_dev_run=True,
            dream_objective_f=datMan.dream_objective_SAE_channel,
        )

    def test_prepare_data(self):
        self.main.prepare_data()
        self.assertIsNotNone(self.main.train_dataset)
        self.assertIsNotNone(self.main.test_dataset)
        self.assertIsNotNone(self.main.train_datasets)
        self.assertIsNotNone(self.main.test_datasets)

        self.assertEqual(len(self.main.train_dataset), 50000) 
        self.assertEqual(len(self.main.test_dataset), 10000)
        self.assertEqual(len(self.main.train_datasets), self.num_tasks)
        self.assertEqual(len(self.main.test_datasets), self.num_tasks)
        for i in range(self.num_tasks):
            self.assertEqual(len(self.main.train_datasets[i]), 50000 / self.num_tasks)
            self.assertEqual(len(self.main.test_datasets[i]), 10000 / self.num_tasks)

    def setup_task_index_prepare(self):
        self.main.prepare_data()

        self.assertIsNone(self.main.current_task_index)
        self.assertIsNone(self.main.train_task)
        self.assertIsNone(self.main.dream_dataset_current_task)

    def setup_task_index(self, idx):
        self.main.setup_task_index(idx)

        self.assertIsNotNone(self.main.current_task_index)
        self.assertIsNotNone(self.main.train_task)
        self.assertIsNone(self.main.dream_dataset_current_task)

        self.assertEqual(self.main.current_task_index, idx)
        self.assertEqual(len(self.main.train_task), 50000 / self.num_tasks)
        tst.assert_close(self.main.train_datasets[idx][0], self.main.train_task[0])

    def test_setup_task_index(self):
        self.setup_task_index_prepare()

        self.setup_task_index(0)
        self.setup_task_index(1)
        self.setup_task_index(2)
        self.setup_task_index(3)
        self.setup_task_index(4)
        with self.assertRaises(Exception):
            self.setup_task_index(5)

    def test_setup_task_index_jumps(self):
        self.setup_task_index_prepare()
        self.setup_task_index(3)
        self.setup_task_index(1)
        with self.assertRaises(Exception):
            self.setup_task_index(5)
        
    def test_split_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = CIFAR10(
            root="data", train=True, transform=transform, download=True
        )
        tasks = [[0,1], [2,3], [4,5], [6,7], [8,9]]
        split_dataset = CLDataModule._split_dataset(train_dataset, tasks)
        for idx, t in enumerate(tasks):
            self.assertEqual(len(split_dataset[idx]), 10000)
        
