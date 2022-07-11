import unittest
from loss_function.chiLoss import ChiLoss
from torch import testing as tst
import torch


class TestChiLoss(unittest.TestCase):
    def setUp(self):
        self.main = ChiLoss(sigma = 0.2, eps=1e-5)

    def test_call_one(self):
        input = torch.tensor([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [2., 2., 2., 2.]
        ]).to('cpu')
        target = torch.tensor([1, 0, 1]).to('cpu')
        # TODO check if this value is correct
        output = torch.tensor(-61.81764602661133).to('cpu')

        ret = self.main(input, target)
        tst.assert_close(ret, output)