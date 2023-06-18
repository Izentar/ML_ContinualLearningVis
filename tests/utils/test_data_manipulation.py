import unittest
from numpy import random
from lucent.optvis import objectives
import math
import numpy as np
from lucent.optvis.objectives import wrap_objective, handle_batch
from torch.utils.data import Subset, random_split, ConcatDataset
import torch
from utils import data_manipulation as datMan
import pytorch_lightning as pl
from torch import testing as tst
from latent_dreams import getDataset, data_transform
from torchvision import transforms

class TestDataManipulation(unittest.TestCase):
    def test_classic_tasks_split_full(self):
        num_classes = 10
        num_tasks = 5
        out = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        ret = datMan.task_split_classic(num_classes, num_tasks)
        self.assertEqual(ret, out)

    def test_classic_tasks_split_part(self):
        num_classes = 5
        num_tasks = 2
        out = [[0, 1], [2, 3], [4]]
        ret = datMan.task_split_classic(num_classes, num_tasks)
        self.assertEqual(ret, out)

    def test_classic_tasks_split_overflow(self):
        num_classes = 5
        num_tasks = 6
        with self.assertRaises(Exception):
            ret = datMan.task_split_classic(num_classes, num_tasks)

    def test_decremental_tasks_split_full(self):
        num_classes = 10
        num_tasks = 5
        out = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [6, 7, 8, 9], [8, 9]]
        ret = datMan.task_split_decremental(num_classes, num_tasks)
        self.assertEqual(ret, out)

    def test_decremental_tasks_split_part(self):
        num_classes = 5
        num_tasks = 2
        out = [[0, 1, 2, 3, 4], [2, 3, 4]]
        ret = datMan.task_split_decremental(num_classes, num_tasks)
        self.assertEqual(ret, out)

    def test_decremental_tasks_split_overflow(self):
        num_classes = 5
        num_tasks = 4
        with self.assertRaises(Exception):
            ret = datMan.task_split_decremental(num_classes, num_tasks)

    def test_classic_select_tasks(self):
        tasks = [[0, 1, 2, 3, 4], [2, 3, 4]]

        self.assertEqual(datMan.select_task_classic(tasks, 0), set([0, 1, 2, 3, 4]))
        self.assertEqual(datMan.select_task_classic(tasks, 1), set([2, 3, 4]))
        with self.assertRaises(Exception):
            datMan.select_task_classic(tasks, 3), set([2, 3, 4])

    def test_decremental_select_tasks(self):
        tasks = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [6, 7, 8, 9], [8, 9]]

        self.assertEqual(datMan.select_task_decremental(tasks, 0), set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertEqual(datMan.select_task_decremental(tasks, 1), set([0, 1]))
        self.assertEqual(datMan.select_task_decremental(tasks, 2), set([2, 3]))
        self.assertEqual(datMan.select_task_decremental(tasks, 3), set([4, 5]))
        self.assertEqual(datMan.select_task_decremental(tasks, 4), set([6, 7]))
        with self.assertRaises(Exception):
            datMan.select_task_decremental(tasks, 5), set([2, 3, 4])

    def test_with_memory_select_tasks(self):
        tasks = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [6, 7, 8, 9], [8, 9]]

        self.assertEqual(datMan.select_task_with_memory(tasks, 0), set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertEqual(datMan.select_task_with_memory(tasks, 1), set([0, 1]))
        self.assertEqual(datMan.select_task_with_memory(tasks, 2), set([0, 1, 2, 3]))
        self.assertEqual(datMan.select_task_with_memory(tasks, 3), set([0, 1, 2, 3, 4, 5]))
        self.assertEqual(datMan.select_task_with_memory(tasks, 4), set([0, 1, 2, 3, 4, 5, 6, 7]))
        with self.assertRaises(Exception):
            datMan.select_task_with_memory(tasks, 5), set([2, 3, 4])

    def test_with_fading_memory_select_tasks(self):
        pl.seed_everything(42)
        #tasks = [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        #tasks = [[0, 1], [0, 1, 2, 3], [4, 5], [4, 5, 6, 7], [8, 9]]
        tasks = [[0, 1], [1, 2, 3], [3, 4, 5], [4, 5, 6], [1, 2, 3, 5, 6, 7, 8, 9]]
        #tasks = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [6, 7, 8, 9], [8, 9]]

        self.assertEqual(datMan.select_task_with_fading_memory(tasks, 0, 1/2), set([0, 1]))
        self.assertEqual(datMan.select_task_with_fading_memory(tasks, 1, 1/2), set([2, 3]))
        self.assertEqual(datMan.select_task_with_fading_memory(tasks, 2, 1/2), set([3, 4, 5]))
        self.assertEqual(datMan.select_task_with_fading_memory(tasks, 3, 1/2), set([6]))
        self.assertEqual(datMan.select_task_with_fading_memory(tasks, 4, 1/2), set([2, 3, 5, 6, 7, 8, 9]))
        with self.assertRaises(Exception):
            datMan.select_task_with_fading_memory(tasks, 5, 1/2)

    def test_normall_dist_tasks_processing_1(self):
        pl.seed_everything(42)
        tasks = [0, 1]
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor(1/2)
        out = torch.tensor([1.1683, 2.0644])

        tst.assert_close(datMan.normall_dist_tasks_processing(tasks, mean=mean, std=std), out, rtol=1e-04, atol=1e-08)

    def test_normall_dist_tasks_processing_2(self):
        pl.seed_everything(42)
        tasks = [0, 1]
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([1/2, 1/3, 1/4])
        out = torch.tensor([1.1683452129364014, 2.0429365634918213])

        tst.assert_close(datMan.normall_dist_tasks_processing(tasks, mean=mean, std=std), out)

    def test_get_target_from_dataset_subset(self):
        dataset = getDataset('c10')(root="data", train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        dat = torch.utils.data.Subset(dataset, range(100))
        targets = datMan.get_target_from_dataset(dat)
        targets_tensor = datMan.get_target_from_dataset(dat, toTensor=True)
        actual_targets = dataset.targets[:100]
        self.assertEqual(type(targets), type(list()))
        self.assertEqual(type(targets_tensor), type(torch.tensor(1)))
        for a, b, c in zip(targets, actual_targets, targets_tensor):
            tst.assert_close(a, b)
            tst.assert_close(torch.tensor(b), c)

    def test_get_target_from_dataset_dataset(self):
        dataset = getDataset('c10')(root="data", train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
        targets = datMan.get_target_from_dataset(dataset)
        targets_tensor = datMan.get_target_from_dataset(dataset, toTensor=True)
        actual_targets = dataset.targets[:100]
        self.assertEqual(type(targets), type(list()))
        self.assertEqual(type(targets_tensor), type(torch.tensor(1)))
        for a, b, c in zip(targets, actual_targets, targets_tensor):
            tst.assert_close(a, b)
            tst.assert_close(torch.tensor(b), c)
            