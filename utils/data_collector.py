import csv
import torch
import numpy as np

def collect_data(collect_f, size, namepath, header:list=None, mode='a'):
    with open(namepath, mode) as f:
        writer = None

        def setup_writer(data):
            if(isinstance(data, dict)):
                if(writer is None):
                    writer = csv.DictWriter(f, mode)
                    if(header is not None):
                        writer.writeheader(header)
            else:
                if(writer is None):
                    writer = csv.writer(f, mode)
                    if(header is not None):
                        writer.writerow(header)
            return writer

        for idx in range(size):
            point = collect_f(idx)
            if(writer is None):
                writer = setup_writer(point)

            if(isinstance(point, torch.TensorType)):
                writer.writerow(point.cpu().detach().numpy())
            elif(isinstance(point, np.ndarray)):
                writer.writerow(point)
            else:
                writer.writerow(np.asarray(point))