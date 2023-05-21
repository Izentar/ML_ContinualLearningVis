import torch

class CyclicBufferByClass(torch.nn.Module):
    def __init__(self, num_classes, dimensions, size_per_class=200, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.size_per_class = size_per_class
        self.dimensions = dimensions
        self.select_flag = torch.tensor(1)
        self.device = device

        tupl = (num_classes, size_per_class, dimensions)
        if(isinstance(dimensions, tuple) or isinstance(dimensions, list)):
            tupl = num_classes + (size_per_class + dimensions)

        self.cyclic_buff = torch.zeros(
            tupl,
            dtype=torch.float32
        ).to(device)

        # first dim - index, second dim - flag if full
        self._buff_idx = torch.zeros((num_classes, 2), dtype=torch.int32).to(device)

    def to(self, device):
        self.cyclic_buff = self.cyclic_buff.to(device)
        self._buff_idx = self._buff_idx.to(device)
        self.select_flag = self.select_flag.to(device)
        return self

    def push(self, cl_idx, val):
        cl_idx = cl_idx.detach().item()
        val = val.detach()
        self.cyclic_buff[cl_idx, self._buff_idx[cl_idx][0]] = val
        self._buff_idx[cl_idx][0] += 1
        if(self._buff_idx[cl_idx][0] >= self.size_per_class):
            self._buff_idx[cl_idx][0] = 0
            self._buff_idx[cl_idx][1] = 1

    def push_target(self, vals, target):
        vals = vals.detach()
        target = target.detach()
        unique = torch.unique(target)
        for u in unique:
            selected = vals[target == u]
            for s in selected:
                self.push(cl_idx=u, val=s)

    def mean_cl(self, cl_idx):
        if(self._buff_idx[cl_idx][1]):
            return torch.mean(self.cyclic_buff[cl_idx], dim=(0, 1))
        return torch.mean(self.cyclic_buff[cl_idx, : self.__get_idx(cl_idx)], dim=(0, 1))

    def __get_idx(self, cl_idx):
        assert isinstance(cl_idx, (int, torch.IntType)), f'Bad index type: {type(cl_idx)}\nValue: {cl_idx}'
        # return current index. Index of the data that was recently pushed
        current_idx = self._buff_idx[cl_idx][0]
        return current_idx - 1 if current_idx != 0 else self.size_per_class - 1

    def _operation_template(self, f):
        #TODO different types returned

        # at the beginning mean can be missleading, not fully filled
        buf = {}
        for cl_idx, flag in enumerate(self._buff_idx):
            flag = flag[1]
            if(flag):
                result = f(self.cyclic_buff[cl_idx])
            else:
                result = f(self.cyclic_buff[cl_idx, : self.__get_idx(cl_idx)])
            buf[cl_idx] = result
        return buf

    def _operation_template_target(self, f, target):
        #TODO different types returned

        # at the beginning mean can be missleading, not fully filled
        flag = self._buff_idx[target][1]
        idx = target
        if(flag):
            result = f(self.cyclic_buff[idx])
        else:
            result = f(self.cyclic_buff[idx, : self.__get_idx(idx)])
        return result

    def mean(self) -> dict[int, torch.Tensor]:
        def __mean(buff, dim=0):
            return torch.mean(buff, dim=dim)
        return self._operation_template(__mean)

    def mean_target(self, target) -> dict[int, torch.Tensor]:
        def __mean(buff, dim=0):
            return torch.mean(buff, dim=dim)
        return self._operation_template_target(__mean, target)

        ## if all flags are on
        ##print(torch.arange(0, self.num_classes).size(), self._buff_idx.size())
        #if torch.all(torch.index_select(self._buff_idx, 0, self.select_flag)):
        #    return torch.mean(self.cyclic_buff, dim=0)
        #
        ## if not every buffer is filled 
        #buf = {}
        #for cl_idx, (_, flag) in zip(range(self.num_classes), self._buff_idx):
        #    if(flag):
        #        mean = torch.mean(self.cyclic_buff[cl_idx], dim=0)
        #    else:
        #        mean = torch.mean(self.cyclic_buff[cl_idx, : self._buff_idx[cl_idx][0]], dim=0)
        #    buf[cl_idx] = mean
        #return buf

    def std(self) -> dict[int, torch.Tensor]:
        def __std(buff, dim=0):
            return torch.std(buff, dim=dim)
        return self._operation_template(__std)

    def std_target(self, target):
        def __std(buff, dim=0):
            return torch.std(buff, dim=dim)
        return self._operation_template_target(__std, target)

    def std_mean(self) -> dict[int, torch.Tensor]:
        def __std_mean(buff, dim=0):
            return torch.std_mean(buff, dim=dim)
        return self._operation_template(__std_mean)

    def std_mean_target(self, target):
        def __std_mean(buff, dim=0):
            return torch.std_mean(buff, dim=dim)
        return self._operation_template_target(__std_mean, target)

    def cov(self) -> dict[int, torch.Tensor]:
        def __cov(buff):
            # https://pytorch.org/docs/stable/generated/torch.cov.html
            buff = torch.transpose(buff, 0, 1)
            return torch.cov(buff)
        return self._operation_template(__cov)

    def cov_target(self, target):
        def __cov(buff):
            # https://pytorch.org/docs/stable/generated/torch.cov.html
            buff = torch.transpose(buff, 0, 1)
            return torch.cov(buff)
        return self._operation_template_target(__cov, target)

    def front(self, target):
        last_idx = self.__get_idx(target)
        return self.cyclic_buff[target, last_idx, :]

    def __len__(self):
        return self.size_per_class

    def get(self, target, idx):
        last_idx = self.__get_idx(target)
        new_idx = (last_idx + idx) % self.size_per_class
        return self.cyclic_buff[target, new_idx, :]
