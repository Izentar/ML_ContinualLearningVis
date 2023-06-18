import torch
import numpy as np
import pandas as pd

def collect_data(collect_f, size, namepath, header:list=None, mode='a'):
    df = pd.DataFrame(columns=header)

    for idx in range(1, size + 1):
        point = collect_f(idx)

        if(isinstance(point, torch.Tensor)):
            tmp = point.cpu().detach().numpy()
        elif(isinstance(point, np.ndarray)):
            tmp = point
        else:
            tmp = np.asarray(point)
        df.loc[idx] = tmp

    df.to_csv(namepath, sep=';', mode=mode, index=False)