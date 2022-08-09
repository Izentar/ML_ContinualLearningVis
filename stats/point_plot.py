import torch
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pickle
import itertools
import json
import os
import pathlib
import sys
from config.default import markers, colors, colors_list

def tryCreatePath(name):
        path = pathlib.Path(name).parent.resolve().mkdir(parents=True, exist_ok=True)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def legend_fix(label_plot: dict, ax):
    label = []
    plot = []
    for cl, val in label_plot.items():
        l = val['label']
        p = val['plot']
        if(isinstance(l, list)):
            label.extend(l)
        else:
            label.append(l)

        if(isinstance(p, list)):
            plot.extend(p)
        else:
            plot.append(p)
    
    for p, l in zip(plot, label):
        p.set_label(l)

    #ax.legend(handles=plot, labels=label)
    ax.legend(bbox_to_anchor = (1.05, 0.6), loc='upper left')

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

                buffer.append((out.detach().to('cpu'), target.detach().to('cpu')))
                counter += dataloader.batch_size
            #print(f'Collect epoch: {epoch}')
            #sys.stdout.flush()

        return buffer

    @staticmethod
    def _cat_buffer(buffer):
        new_buffer_batch = []
        new_buffer_target = []
        for batch, target in buffer:
            new_buffer_batch.append(batch)
            new_buffer_target.append(target)

        new_buffer_batch = torch.cat(new_buffer_batch, dim=0)
        new_buffer_target = torch.cat(new_buffer_target, dim=0)

        return new_buffer_batch, new_buffer_target

    @staticmethod
    def by_class_operation(f, buffer, fileName):
        '''
            buffer - tuple(batch, target)
        '''

        new_buffer_batch, new_buffer_target = Statistics._cat_buffer(buffer)
        unique, unique_count = torch.unique(new_buffer_target, return_counts=True)
        output = {}

        # loop buffer
        tryCreatePath(fileName)
        with open(fileName, 'w') as file:
            for cl, cl_count in zip(unique, unique_count):
                cl_indices = torch.isin(new_buffer_target, cl)
                cl_indices_list = torch.where(cl_indices)[0]
                cl_batch = torch.index_select(new_buffer_batch, 0, cl_indices_list)

                file.write(f'-------------------------\nClass {cl}, count {cl_count}\n')
                f(cl, cl_count, output, cl_batch, new_buffer_batch, new_buffer_target, file)
        return output

    @staticmethod
    def f_mean_std(cl, cl_count, output: dict, cl_batch, buffer_batch, buffer_target, file):
        '''
            Returns in output
            [std, mean]
        '''
        std, mean = torch.std_mean(cl_batch, dim=0)
        output[cl] = {
            'std': std,
            'mean': mean,
        }
        file.write(f"Mean {mean}\nstd {std}\n")

    @staticmethod
    def mean_std2(buffer, fileName='saves/mean_std.txt'):
        '''
            buffer - tuple(batch, target)
        '''

        new_buffer_batch, new_buffer_target = Statistics._cat_buffer(buffer)
        unique, unique_count = torch.unique(new_buffer_target, return_counts=True)
        std_mean_dict = {}

        # loop buffer
        tryCreatePath(fileName)
        with open(fileName, 'w') as file:
            for cl, u_count in zip(unique, unique_count):
                file.write(f'-------------------------\nClass {cl}, count {u_count}\n')
                cl_indices = torch.isin(new_buffer_target, cl)
                cl_indices_list = torch.where(cl_indices)[0]

                cl_batch = torch.index_select(new_buffer_batch, 0, cl_indices_list)
                std, mean = torch.std_mean(cl_batch, dim=0)
                std_mean_dict[cl] = {
                    'std': std,
                    'mean': mean,
                }
                file.write(f"Mean {mean}\nstd {std}\n")
        return std_mean_dict

    @staticmethod
    def f_distance(cl, cl_count, output, cl_batch, buffer_batch, buffer_target, file):
        '''
            Returns in output
            {
                'std': std,
                'mean': mean,
                'distance_positive': distance_positive,
                'distance_negative': distance_negative,
            }
        '''
        Statistics.f_mean_std(cl, cl_count, output, cl_batch, buffer_batch, buffer_target, file)
        pdist = torch.nn.PairwiseDistance(p=2)

        mean = torch.unsqueeze(output[cl]['mean'], 0)
        distance_positive = pdist(cl_batch, mean)
        output[cl]['distance_positive'] = distance_positive

        cl_batch_set = set(cl_batch)
        buffer_batch_set = set(buffer_batch)
        negative_batch = list(buffer_batch_set - cl_batch_set)

        distance_negative = pdist(torch.stack(negative_batch), mean)

        output[cl]['distance_negative'] = distance_negative
        


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

    def flush(self, fig, ax, name, show, idx=None, ftype='svg'):
        ax.legend()
        ax.grid(True)
        if(show):
            plt.show()
        if(name is not None):
            if(idx is None):
                fig.savefig(f"{name}.{ftype}")
            else:
                print(f"{name}_{idx}")
                fig.savefig(f"{name}_{idx}.{ftype}")

    def plot_distance(
        self, 
        std_mean_distance_dict, 
        name='point-distance', 
        show=False, 
        ftype='png',
        markersize=2,
    ):
        my_colors = colors_list[:2]
        legend_label = {}

        for idx, (cl, val) in enumerate(std_mean_distance_dict.items()):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            mean = val['mean']
            dist_positive = val['distance_positive']
            dist_negative = val['distance_negative']
            my_colors = itertools.cycle(my_colors)
            
            for pos, neg in zip(dist_positive, dist_negative):
                theta = np.random.uniform(0, 2 * np.pi)
                color = next(my_colors)
                pos_plot = ax.plot(theta, pos, 'ro', color=color, markersize=markersize)[0]
                theta = np.random.uniform(0, 2 * np.pi)
                color = next(my_colors)
                neg_plot = ax.plot(theta, neg, 'ro', color=color, markersize=markersize)[0]
                legend_label[cl] = {
                    'label': [f'Positive {cl}', f'Negative {cl}'], 
                    'plot': [pos_plot, neg_plot],
                }
            ax.set_title(f'Class {cl}')
            legend_fix(legend_label, ax)
            self.flush(fig, ax, name, show, idx=idx, ftype=ftype)

    def plot_std_mean(
        self, 
        std_mean_dict, 
        name='std-mean', 
        show=False, 
        ftype='png'
    ):
        fig, ax = plt.subplots()
        my_colors = colors_list[:len(std_mean_dict)]
        my_colors = itertools.cycle(my_colors)

        mini = 0
        maxi = 0
        border_range_scale = 0.2
        ellipse_height = 0.75
        legend_label = {}

        for cl, val in std_mean_dict.items():
            std = val['std']
            mean = val['mean']
            mini = min(mean - std)
            maxi = max(mean + std)
            labels = [f'dim-{x}' for x in range(len(mean))]
            plot_index = 0

            # iterate over mean and std of dimensions
            for idx1, (m1, s1) in enumerate(zip(mean, std)):
                color = next(my_colors)
                new_plot = ax.plot(m1, plot_index, 'ro', color=color)[0]
                legend_label[cl] = {
                    'label': [f'Class {cl}'], 
                    'plot': [new_plot],
                }
                ax.add_artist(Ellipse((m1.item(), plot_index), s1, ellipse_height, color=color, fill=False))
                  
                plot_index += 1
                #if(idx1 + 1 < len(mean)):
                #    ax.axhline(y=plot_index - 0.5, color='black')
        legend_fix(legend_label, ax)
        ax.set_title('std-mean')
        plt.yticks(list(range(len(labels))), labels, rotation='horizontal')
        add_range = (maxi - mini) * border_range_scale
        plt.xlim([mini - add_range, maxi + add_range])
        self.flush(fig, ax, name, show, ftype=ftype)

    def plot(
        self, 
        buffer, 
        plot_type, 
        with_batch=True, 
        with_target=True, 
        symetric=True, 
        name='point-plot', 
        show=False, 
        markersize=1, 
        ftype='svg'
    ):
        '''
            buffer[0] - list of batches of points
            buffer[1] - list of points
            buffer[2:] -  points
        '''
        if not (plot_type in ['singular', 'multi']):
            raise Exception(f"Unknown plot type: {plot_type}")

        tryCreatePath(name)

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

        def plot_loop(data_x_target, data_y_target, data_dims, markersize):
            fig, ax = plt.subplots()
            for (kx, vx), (ky, vy), (kt, vt) in zip(data_x_target.items(), data_y_target.items(), data_dims.items()):
                ax.plot(
                    vx,
                    vy,
                    'ro', 
                    marker=next(markers),
                    color=next(colors),
                    label=f"Target: {kx}",
                    markersize=markersize,
                )
            return fig, ax

        if(plot_type == 'singular'):
            for idx, (data_x_target, data_y_target, data_dims) in enumerate(stash):
                fig, ax = plot_loop(data_x_target, data_y_target, data_dims, markersize)
                self.flush(fig, ax, name, show, idx=idx, ftype=ftype)
        elif(plot_type == 'multi'):
            fig, ax = plot_loop(data_x_target, data_y_target, data_dims, markersize)
            self.flush(fig, ax, name, show, ftype=ftype)

    def saveBuffer(self, buffer, name):
        tryCreatePath(name)

        newb = []
        for (a, b) in buffer:
            newb.append((a.cpu().detach().numpy().tolist(), b.cpu().detach().numpy().tolist()))

        with open(f'{name}.pickle', 'wb') as fp:
            pickle.dump(newb, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'{name}.json', 'w') as fp:
            json.dump(newb, fp)

    def loadBuffer(self, name):
        buffer = None
        if('.json' in name):
            with open(name, 'r') as fp:
                buffer = json.load(fp)
        elif('.pickle' in name):
            with open(name, 'rb') as fp:
                buffer = pickle.load(fp)

        newb = []
        for (a, b) in buffer:
            newb.append((torch.from_numpy(np.asarray(a)), torch.from_numpy(np.asarray(b))))
        return newb

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
    plotter.plot([(x1, [1, 3, 5]), (x2, [7, 5, 3])], plot_type='singular', show=True, name='plots/point-plot')