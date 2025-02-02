import torch
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import pickle
import itertools
import json
import os
import pathlib
from pathlib import Path
import sys
from config.default import markers, colors, colors_list
from utils.data_manipulation import select_class_indices_tensor
import torchmetrics

import wandb

def tryCreatePath(name):
        path = pathlib.Path(name).parent.resolve().mkdir(parents=True, exist_ok=True)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def set_legend(label_plot: dict, ax):
    label = []
    plot = []
    for _, val in label_plot.items():
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

    ax.legend()

def classes_to_int(classes):
    if classes is not None:
        return [int(i) for i in classes]
    return classes

def class_not_present(classes, x):
    if not (classes is None or len(classes) == 0):
        if (isinstance(x, torch.Tensor) and x.item() not in classes) or (isinstance(x, int) and x not in classes):
            return True
    return False

class ServePlot():
    def __init__(self, **kwargs):
        self.create_new_subplot = True
        self.kwargs = kwargs
        self.nrows = kwargs.get('nrows', 1)
        self.ncols = kwargs.get('ncols', 1)
        self.index_list = [(x, y) for x in range(self.nrows) for y in range(self.ncols)]
        self.axs = None

        self.current_fig = None
        self.current_ax = None

        self.del_flush = True
        self.name = None

    def get_next(self, to_wandb):
        if(self.create_new_subplot or self.axs is None):
            self.create_new_subplot = False
            self.ax_idx = 0
            self.fig, self.axs = plt.subplots(**self.kwargs)

            self.fig.set_size_inches(6.4 * self.nrows, 4.8 * self.ncols)
            if(self.name is not None):
                self._flush(to_wandb=to_wandb)

        ax = self.axs
        if(isinstance(self.axs, list) or isinstance(self.axs, np.ndarray)):
            # bad size of nrow, ncol if exception thrown
            ax = self.axs[self.index_list[self.ax_idx]]
            self.ax_idx += 1
            if(self.ax_idx == len(self.index_list)):
                self.create_new_subplot = True
        self.current_ax = ax
        self.current_fig = self.fig
        return self.fig, ax

    def schedule_flush(self, plot_point_obj, name, show, ftype, idx=None):
        self.name = name
        self.show = show
        self.idx = idx
        self.ftype = ftype
        self.plot_point_obj = plot_point_obj

    def _flush(self, to_wandb):
        if(self.name is None):
            raise Exception('Cannot flush to file. No config data provided.')
        self.plot_point_obj.flush(self.current_fig, self.current_ax, self.name, self.show, idx=self.idx, ftype=self.ftype, to_wandb=to_wandb)
        self.name = None

    def __del__(self):
        if(self.del_flush):
            self._flush()

    def force_flush(self, plot_point_obj, name, show, ftype, to_wandb, idx=None):
        self.del_flush = False
        self.schedule_flush(plot_point_obj, name=name, show=show, idx=idx, ftype=ftype)
        self._flush(to_wandb)


class Statistics():
    def __init__(self):
        pass

    def collect(self, model, dataloader, num_of_points, to_invoke, num_classes, logger=None):
        '''
            Collect given num_of_points to use them on calculating stats and to plot. 
            to_invoke: function to invoke that returns points to collect
        '''
        buffer = []
        model.eval()
        batch_size = len(next(iter(dataloader))[1])
        input_device = next(iter(dataloader))[1][0].device
        epoch_size = np.maximum(num_of_points // len(dataloader) // batch_size, 1)
        counter = 0
        if logger is not None:
            model.loss_to(input_device)

        accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

        with torch.no_grad():
            for epoch in range(epoch_size):
                # iterate over dataset points one at the time 
                for idx, (input, target) in enumerate(dataloader):
                    input = input.to(model.device)
                    target = target.to(model.device)
                    if(counter >= num_of_points):
                        break
                    out = to_invoke(model, input)

                    if logger is not None:
                        loss = model.call_loss(out, target, train=False)
                        logger.log_metrics({f'stats/collect_loss': loss}, counter)

                        predictions: torch.Tensor = model.classify(out)
                        accuracy(predictions, target)
                        logger.log_metrics({f'stats/collect_accuracy': accuracy.compute().item()}, counter)


                    buffer.append((out.detach().to('cpu'), target.detach().to('cpu')))
                    counter += batch_size

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
    def by_class_operation(f, buffer, fileName, output:dict=None):
        '''
            buffer - tuple(batch, target)
        '''

        new_buffer_batch, new_buffer_target = Statistics._cat_buffer(buffer)
        unique, unique_count = torch.unique(new_buffer_target, return_counts=True, dim=0)
        if(output is None):
            output = {}

        # loop buffer
        tryCreatePath(fileName)
        with open(fileName, 'w') as file:
            for cl, cl_count in zip(unique, unique_count):
                cl_batch = new_buffer_batch[new_buffer_target == cl]

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
        output[cl.item()] = {
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
        unique, unique_count = torch.unique(new_buffer_target, return_counts=True, dim=0)
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

        mean = output[cl.item()]['mean']
        mean_positive = mean.repeat(len(cl_batch), 1)
        distance_positive = pdist(cl_batch, mean_positive)
        output[cl.item()]['distance_positive'] = distance_positive

        combined = torch.cat((cl_batch, buffer_batch))
        uniqueness, combined_count = combined.unique(return_counts=True, dim=0)
        negative_batch = uniqueness[combined_count == 1]

        mean_negative = mean.repeat(len(negative_batch), 1)
        distance_negative = pdist(negative_batch, mean_negative)

        output[cl.item()]['distance_negative'] = distance_negative

    @staticmethod
    def f_average_point_dist_from_means(cl, cl_count, output, cl_batch, buffer_batch, buffer_target, file):
        '''
            Calculate distance between means and current cl_batch. Takes average of the result.
            Needs from output values of 'mean'.
            Returns in output
            {
                'average_point_dist_from_means': {class_2: average_point_dist_from_means},
            }
        '''

        means = []
        pdist = torch.nn.PairwiseDistance(p=2)
        cl = cl.item()

        file.write(f'Avg distance of points of the class {cl}\n')

        # for given cl iterate over output in search for means.
        # calculate distance from cl2 mean to all current points in cl_batch and average result
        for cl2, val in output.items():
            mean = val['mean']

            if (not 'average_point_dist_from_means' in output[cl2].keys()) or \
                (not isinstance(output[cl2]['average_point_dist_from_means'], dict)):
                output[cl2]['average_point_dist_from_means'] = {}

            #indices = select_class_indices_tensor(cl, buffer_target)
            #cl_batch_latent = buffer_batch[indices]
            #cl_batch_latent = buffer_batch[buffer_target == cl]
            cl_batch_latent = cl_batch
            mean = mean.repeat(len(cl_batch_latent), 1)
            distance_batch = pdist(cl_batch_latent, mean)
            calculated_dist_mean = torch.mean(distance_batch, dim=0)
            file.write(f'{cl2}\t')
            file.write(f'{calculated_dist_mean.numpy()}\n')

            output[cl2]['average_point_dist_from_means'][cl] = calculated_dist_mean

    @staticmethod
    def mean_distance(output):
        '''
            Calculate distance between means.
            It returns at least on the output
            {
                'mean': mean,
                'distance_mean': distance_mean,
            }
        '''

        pdist = torch.nn.PairwiseDistance(p=2)

        for cl, val in output.items():
            means = []
            main_mean = None
            for cl2, val2 in output.items():
                if(cl == cl2):
                    main_mean = val2['mean']
                    continue
                means.append(val2['mean'])
            other_means = torch.stack(means, 0)
            main_mean = main_mean.repeat(len(other_means), 1)
            distance_mean = pdist(other_means, main_mean)
            output[cl]['distance_mean'] = distance_mean
        
        return output

class PointPlot():
    def __init__(self, root='plots'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _try_create_dir(path_with_filename):
        if(not isinstance(path_with_filename, Path)):
            path_with_filename = Path(path_with_filename)
        path_with_filename.parent.mkdir(exist_ok=True, parents=True)

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
        
    def flush(self, fig, ax, name, show, to_wandb, idx=None, ftype='png'):
        tryCreatePath(name)
        ax.legend()
        if(name is not None):
            n = None
            if(idx is None):
                n = f"{name}.{ftype}"
            else:
                full_name = f'{name}_idx{idx}'
                print(f"INFO: Plot {full_name}")
                n = f"{full_name}.{ftype}"
            fig.savefig(n)
            if(to_wandb):
                wandb.log({f"custom_plots/{n}": wandb.Image(fig)})
        if(show):
            plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def plot_distance(
        self, 
        std_mean_distance_dict, 
        name='point-distance', 
        show=False, 
        ftype='png',
        markersize=2,
        nrows=1,
        ncols=1,
        alpha=0.4,
        to_wandb=True,
    ):
        my_colors = colors_list[:2]
        legend_label = {}

        create_new_subplot = True
        plotter = ServePlot(nrows=nrows, ncols=ncols, subplot_kw={'projection': 'polar'})
        
        for idx, (cl, val) in enumerate(std_mean_distance_dict.items()):
            fig, ax = plotter.get_next(to_wandb=to_wandb)

            dist_positive = val['distance_positive']
            dist_negative = val['distance_negative']
            my_colors = itertools.cycle(my_colors)        

            theta = [np.random.uniform(0, 2 * np.pi) for _ in range(len(dist_negative))]
            color = next(my_colors)
            neg_plot = ax.plot(theta, dist_negative, 'o', color=color, markersize=markersize, alpha=alpha)[0]

            # second plot to not obscure the minor positive class
            theta = [np.random.uniform(0, 2 * np.pi) for _ in range(len(dist_positive))]
            color = next(my_colors)
            pos_plot = ax.plot(theta, dist_positive, 'o', color=color, markersize=markersize, alpha=alpha)[0]

            legend_label[cl] = {
                'label': [f'Positive {cl}', f'Negative {cl}'], 
                'plot': [pos_plot, neg_plot],
            }

            
            #for pos, neg in zip(dist_positive, dist_negative):
            #    theta = np.random.uniform(0, 2 * np.pi)
            #    color = next(my_colors)
            #    pos_plot = ax.plot(theta, pos, 'o', color=color, markersize=markersize)[0]
            #    theta = np.random.uniform(0, 2 * np.pi)
            #    color = next(my_colors)
            #    neg_plot = ax.plot(theta, neg, 'o', color=color, markersize=markersize)[0]
            #    legend_label[cl] = {
            #        'label': [f'Positive {cl}', f'Negative {cl}'], 
            #        'plot': [pos_plot, neg_plot],
            #    }
            ax.set_title(f'Class {cl}')
            plotter.schedule_flush(self, name=name, show=show, idx=idx, ftype=ftype)
            set_legend(legend_label, ax)
            #self.flush(fig, ax, name, show, idx=idx, ftype=ftype)

        plotter.force_flush(self, name=name, show=show, idx=idx+1, ftype=ftype, to_wandb=to_wandb)

    def plot_std_mean(
        self, 
        std_mean_dict, 
        name='std-mean', 
        show=False, 
        ftype='png',
        to_wandb=True,
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
            color = next(my_colors)
            labels = [f'dim-{x}_color-{str(color)}' for x in range(len(mean))]
            plot_index = 0

            # iterate over dimensions of mean and std
            for idx1, (m1, s1) in enumerate(zip(mean, std)):
                new_plot = ax.plot(m1, plot_index, 'o', color=color)[0]
                legend_label[cl] = {
                    'label': [f'Class {cl}'], 
                    'plot': [new_plot],
                }
                ax.add_artist(Ellipse((m1.item(), plot_index), s1, ellipse_height, color=color, fill=False))
                  
                plot_index += 1
                #if(idx1 + 1 < len(mean)):
                #    ax.axhline(y=plot_index - 0.5, color='black')
        set_legend(legend_label, ax)
        ax.set_title('std-mean')
        plt.yticks(list(range(len(labels))), labels, rotation='horizontal')
        add_range = (maxi - mini) * border_range_scale
        plt.xlim([mini - add_range, maxi + add_range])
        self.flush(fig, ax, name, show, idx=len(std_mean_dict), ftype=ftype, to_wandb=to_wandb)

    def plot_mean_distance(
        self, 
        mean_distance_dict, 
        name='mean-distance', 
        show=False, 
        ftype='png',
        markersize=2,
        to_wandb=True
    ):
        fig, ax = plt.subplots()
        legend_label = {}
        my_colors = colors_list[:len(mean_distance_dict)]
        my_colors = itertools.cycle(my_colors)

        y_labels = []
        
        for idx, (cl, val) in enumerate(mean_distance_dict.items()):
            main_mean_dist = val['distance_mean']
            other_mean_dist = []
            other_class_dist = []
            y_labels.append(cl)

            #for idx2, (cl2, val2) in enumerate(mean_distance_dict.items()):
            #    mean_dist = val2['distance_mean']
            #    if(idx != idx2):
            #        other_mean_dist.append(mean_dist)
            #        other_class_dist.append(cl2)
            #other_mean_dist = torch.stack(other_mean_dist, 0)
            #other_class_dist = torch.stack(other_class_dist, 0)

            #for points in main_mean_dist:
            color = next(my_colors)
                #theta = [np.random.uniform(0, 2 * np.pi) for _ in range(len(other_mean_dist))]
                #theta = np.random.uniform(0, 2 * np.pi)
            y = [idx] * (len(main_mean_dist))
            new_plot = ax.plot(main_mean_dist, y, 'o', color=color, markersize=markersize)[0]
            
            legend_label[cl] = {
                'label': f'Relative to class {cl}', 
                'plot': new_plot,
            }

        set_legend(legend_label, ax)
        ax.set_title(f'Distance of the means from each other')
        plt.yticks(list(range(len(y_labels))), y_labels, rotation='horizontal')
        self.flush(fig, ax, name, show, idx=idx+1, ftype=ftype, to_wandb=to_wandb)

    def plot(
        self, 
        buffer, 
        plot_type, 
        with_batch=True, 
        with_target=True, 
        symetric=True,
        to_wandb=True, 
        classes=None,
        name='point-plot', 
        show=False, 
        markersize=1, 
        ftype='png',
        limit_plots_to=6,
    ):
        '''
            Iterate over vector, choose 2 of them and plot them. Do it for 
            all combinations of the scalars inside vector.
            plot_type - show all plots on one figure or multiple figures as multiple files.

            Depends on the flags with_batch and with_target
            buffer[0] - list of batches of points
            buffer[1] - list of points
            buffer[2, :] -  points
        '''
        if not (plot_type in ['singular', 'multi']):
            raise Exception(f"Unknown plot type: {plot_type}")

        tryCreatePath(name)

        target = None
        if(with_batch and with_target):
            target = [x[1].clone().detach() for x in buffer]
            buffer = [x[0] for x in buffer]
            buffer = torch.cat(buffer, dim=0)
            target = torch.cat(target, dim=0).tolist()
        elif(with_batch):
            if(isinstance(buffer, list)):
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
        counter = 0

        # iterate over x and y of the plot
        for dim_x in range(dims):
            if(limit_plots_to >= counter):
                break
            if(symetric):
                start = 0
            else:
                start = dim_x
            for dim_y in range(start, dims):
                counter += 1
                if(dim_x == dim_y):
                    continue
                if(limit_plots_to >= counter):
                    break
                data_x = torch.stack([x[dim_x].to('cpu') for x in buffer], dim=0).tolist()
                data_y = torch.stack([y[dim_y].to('cpu') for y in buffer], dim=0).tolist()

                for t in target_set:
                    data_x_target[t].extend(np.take(data_x, target_legend[t]).tolist())
                    data_y_target[t].extend(np.take(data_y, target_legend[t]).tolist())
                    data_dims[t].append((dim_x, dim_y))
                if(plot_type == 'singular'):
                    stash.append((data_x_target, data_y_target, data_dims))
                    data_x_target, data_y_target, data_dims = self.create_buffers(target_set)  

        def select_marker_color(data_x_target):
            class_marker_color = {}
            ret = []
            keys = np.unique(list(data_x_target.keys()))
            for k in keys:
                class_marker_color[k] = (next(markers), next(colors))

            for k, v in data_x_target.items():
                ret.append(class_marker_color[k])
            return ret

        def plot_loop(data_x_target, data_y_target, data_dims, markers_colors, markersize, classes):
            fig, ax = plt.subplots()
            classes = classes_to_int(classes)
            for (kx, vx), (ky, vy), (kt, vt), (marker, color) in zip(data_x_target.items(), data_y_target.items(), data_dims.items(), markers_colors):
                if(class_not_present(classes, kx)):
                    continue
                ax.scatter(
                    vx,
                    vy,
                    marker=marker,
                    color=color,
                    label=f"Target: {kx}",
                    s=markersize,
                )
            return fig, ax

        markers_colors = select_marker_color(data_x_target=data_x_target)
        if(plot_type == 'singular'):
            for idx, (data_x_target, data_y_target, data_dims) in enumerate(stash):
                fig, ax = plot_loop(data_x_target, data_y_target, data_dims, markers_colors=markers_colors, markersize=markersize, classes=classes)
                self.flush(fig, ax, name, show, idx=idx, ftype=ftype, to_wandb=to_wandb)
        elif(plot_type == 'multi'):
            fig, ax = plot_loop(data_x_target, data_y_target, data_dims, markers_colors=markers_colors, markersize=markersize, classes=classes)
            self.flush(fig, ax, name, show, ftype=ftype, to_wandb=to_wandb)

    def plot_3d(
        self, 
        buffer,
        std_mean_dict,
        name='point-plot-3d', 
        show=True,
        to_wandb=True,
        classes=None,
        ftype='png',
        alpha=0.3,
        space=30,
    ):
        last_dim = np.shape(buffer[0][0].numpy())[-1]
        if last_dim != 3:
            print(f'\nWARNING: Plot 3D only for 3 dimensional space! Found {last_dim} dimensions.')
            return
        tryCreatePath(name)
        fig, ax = plt.figure(), plt.axes(projection='3d')

        points = []
        target = []
        for b, t in buffer:
            points.append(b)
            target.append(t)

        points = torch.cat(points)
        target = torch.cat(target)

        unique_target, unique_count = torch.unique(target, return_counts=True)

        classes = classes_to_int(classes)
        for cl, count in zip(unique_target, unique_count):
            if(class_not_present(classes, cl)):
                continue
            #mean = std_mean_dict[cl]['mean']

            indices = select_class_indices_tensor(cl, target)
            current_points_class = points[indices]

            swap_points = torch.swapaxes(current_points_class, 1, 0)
            #print(swap_points.size())
            ax.scatter(swap_points[0], swap_points[1], swap_points[2], color=next(colors), marker=next(markers), label=f'Class {cl}', alpha=alpha, s=space)

        self.flush(fig, ax, name, show, ftype=ftype, to_wandb=to_wandb)

    def plot_mean_dist_matrix(
        self, 
        average_point_dist_from_means_dict, 
        name='mean-dist-matrix', 
        show=False, 
        ftype='png',
        size_x=15.4,
        size_y=10.8,
        precision=4,
        to_wandb=True,
    ):
        fig, ax = plt.subplots(figsize=(size_x,size_y))
        formated_matrix = []

        for cl, val in average_point_dist_from_means_dict.items():
            second = val['average_point_dist_from_means']
            inner = []
            for cl2, mean in second.items():
                inner.append(mean.item())
            formated_matrix.append(inner)

        formated_matrix = np.array(formated_matrix)
        ax.matshow(formated_matrix, cmap=plt.cm.Greens, aspect='auto')

        for (i, j), z in np.ndenumerate(formated_matrix):
            ax.text(j, i, np.format_float_positional(z, precision=precision), va='center', ha='center')

        self.flush(fig, ax, name, show, idx=len(average_point_dist_from_means_dict), ftype=ftype, to_wandb=to_wandb)

    def plot_bar(
        self, 
        data:dict, 
        name='bar_plot', 
        show=False, 
        ftype='png',
        size_x=15.4,
        size_y=10.8,
        nrows=1,
        ncols=1,
        to_wandb=True,
    ):
        name = self.root / name
        self._try_create_dir(name)
        plotter = ServePlot(nrows=nrows, ncols=ncols, figsize=(size_x,size_y))
        legend_label = dict()
        for cl, x in data.items():
            x:np.ndarray = x.to('cpu').numpy()
            fig, ax = plotter.get_next(to_wandb=to_wandb)
            plot = ax.bar(range(len(x)), x)

            ax.grid(True)
            legend_label[cl] = {
                'label': [f'Class {cl}'], 
                'plot': [plot],
            }
            set_legend(legend_label, ax)
        
        plotter.force_flush(self, name=name, show=show, ftype=ftype, to_wandb=to_wandb)

    def plot_errorbar(
            self, 
            data:dict, 
            name='bar_plot', 
            show=False, 
            ftype='png',
            size_x=15.4,
            size_y=10.8,
            to_wandb=True,
        ):
            name = self.root / name
            self._try_create_dir(name)
            fig, ax = plt.subplots(figsize=(size_x,size_y))
            legend_label = dict()
            x_min = []
            x_max = []
            x_std = []
            x_mean = []
            classes = []
            for cl, x in data.items():
                x:np.ndarray = x.to('cpu').numpy()
                x_min.append(np.min(x))
                x_max.append(np.max(x))
                x_std.append(np.std(x))
                x_mean.append(np.mean(x))
                classes.append(cl)
            x_min = np.array(x_min)
            x_max = np.array(x_max)
            x_std = np.array(x_std)
            x_mean = np.array(x_mean)
            plot = ax.errorbar(np.arange(len(classes)), x_mean, x_std, fmt='ok', lw=3)
            plot = ax.errorbar(np.arange(len(classes)), x_mean, yerr=[x_mean - x_min, x_max - x_mean], fmt='.k', ecolor='green', lw=1)
            ax.set_xlim(-1, len(classes))

            plot.set_label(f'Min/Max/Mean/STD - Class {classes}')
            self.flush(fig=fig, ax=ax, name=name, show=show, ftype=ftype, to_wandb=to_wandb)

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