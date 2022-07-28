import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from model.overlay import CLModel

class Statistics():
    def __init__(self):
        pass

    def collect(self, model, dataloader, num_of_points, to_invoke):
        '''
            to_invoke: function to invoke that returns points to collect
        '''
        buffer = []
        model.eval()
        epoch_size = np.maximum(num_of_points // len(dataloader) // dataloader.batch_size, 1)
        counter = 0

        for epoch in range(epoch_size):
            for idx, (input, target) in enumerate(dataloader):
                if(counter >= num_of_points):
                    break
                out = to_invoke(model, input)

                buffer.append((out, target))
                counter += dataloader.batch_size

        return buffer


class PointPlot():
    def __init__(self):
        pass

    @staticmethod
    def inverse_dict(d):
        newd = {}
        for k, val in d.items():
            for v in val:
                newd[v] = k
        return newd

    def create_buffers(self, target_set):
        data_x_target = {}
        data_y_target = {}
        data_dims = {}
        for t in target_set:
            data_x_target[t] = []
            data_y_target[t] = []
            data_dims[t] = []
        return data_x_target, data_y_target, data_dims

    def flush(self, fig, ax, name, show, idx=None):
        ax.legend()
        ax.grid()
        if(show):
            plt.show()
        if(name is not None):
            if(idx is None):
                fig.savefig(name)
            else:
                print(f"{name}_{idx}")
                fig.savefig(f"{name}_{idx}.svg")

    def plot(self, buffer, plot_type, with_batch=True, with_target=True, symetric=True, name='point-plot', show=False):
        '''
            buffer[0] - list of batches of points
            buffer[1] - list of points
            buffer[2:] -  points
        '''
        if not (plot_type in ['singular', 'multi']):
            raise Exception(f"Unknown plot type: {plot_type}")

        target = None
        if(with_batch and with_target):
            target = [torch.tensor(x[1]) for x in buffer]
            buffer = [x[0] for x in buffer]
            buffer = torch.cat(buffer, dim=0)
            target = torch.cat(target, dim=0).tolist()
        elif(with_batch):
            buffer = torch.cat(buffer, dim=0)
        elif(with_target):
            target = buffer[1].tolist()
            buffer = buffer[0]

        dims = len(buffer[0])
        target_legend = {}
        target_set = list(set(target))
        for t in target_set:
            task_indices = np.isin(np.array(target), t)
            target_legend[t] = np.where(task_indices)[0]
        markers = itertools.cycle(('>', '+', '.', 'o', '*'))
        colors = itertools.cycle(('r', 'g', 'b', 'c', 'k', 'm', 'y'))

        data_x_target, data_y_target, data_dims = self.create_buffers(target_set)
        stash = []

        for dim_x in range(dims):
            if(symetric):
                start = 0
            else:
                start = dim_x
            for dim_y in range(start, dims):
                if(dim_x == dim_y):
                    continue
                data_x = torch.stack([x[dim_x].to('cpu') for x in buffer], dim=0).tolist()
                data_y = torch.stack([y[dim_y].to('cpu') for y in buffer], dim=0).tolist()

                for t in target_set:
                    data_x_target[t].extend(np.take(data_x, target_legend[t]).tolist())
                    data_y_target[t].extend(np.take(data_y, target_legend[t]).tolist())
                    data_dims[t].append((dim_x, dim_y))
                if(plot_type == 'singular'):
                    stash.append((data_x_target, data_y_target, data_dims))
                    data_x_target, data_y_target, data_dims = self.create_buffers(target_set)  

        def plot_loop(data_x_target, data_y_target, data_dims):
            fig, ax = plt.subplots()
            for (kx, vx), (ky, vy), (kt, vt) in zip(data_x_target.items(), data_y_target.items(), data_dims.items()):
                ax.plot(
                    vx,
                    vy,
                    'ro', 
                    marker=next(markers),
                    color=next(colors),
                    label=f"Target: {kx}"
                )
            return fig, ax

        if(plot_type == 'singular'):
            for idx, (data_x_target, data_y_target, data_dims) in enumerate(stash):
                fig, ax = plot_loop(data_x_target, data_y_target, data_dims)
                self.flush(fig, ax, name, show, idx=idx)
        elif(plot_type == 'multi'):
            fig, ax = plot_loop(data_x_target, data_y_target, data_dims)
            self.flush(fig, ax, name, show)

if __name__ == '__main__':
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

    y1 = torch.tensor([
        [0, 1],
        [4, 5],
        [8, 9],
    ])

    y2 = torch.tensor([
        [12, 13],
        [16, 17],
        [20, 21],
    ])

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

    plotter = PointPlot()
    plotter.plot([(x1, [1, 3, 5]), (x2, [7, 5, 3])], plot_type='singular', show=True)