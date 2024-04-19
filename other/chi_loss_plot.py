import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

def loss(value, k, std, eps=1e-6):
    first_part= -(k / 2 - 1) * np.log(value / std**2 + eps)
    second_part = value / (2 * std**2)
    return first_part + second_part

def loss_k_eq_2(value, k, std, eps=1e-6):
    second_part = value / (2 * std**2)
    return second_part

def loss_2(value, k, std, eps=1e-6):
    value = value / (2 * std**2) 
    first_part= -(k / 2 - 1) * np.log(value / k + eps)
    second_part = value / (2 * k)
    return first_part + second_part

def select_min_max(min_y, max_y, new_min_y, new_max_y):
    if(min_y < new_min_y):
        new_min_y = min_y
    if(max_y > new_max_y):
        new_max_y = max_y
    return new_min_y, new_max_y

def plot_chi_square(f, ax, stop, k, std, start=None, step=None):
    if(step is None):
        step = 1000
    if(start is None):
        start = 0
    x = np.linspace(start, stop, int(step))
    output = list(map(lambda x: f(value=x, std=std, k=k), x))
    if(np.isnan(output).any()):
        raise Exception("Ploted function returned NaN.")
    ax.plot(x, output, label=f'std={std}; k={k}')
    return min(output), max(output)

def plot_chi_square_loop(f, ax, stop, k, std, min_y=None, max_y=None, min_x=None, max_x=None, step=None, start=None):
    check_min_y = 0
    check_max_y = 0
    check_min_x = None
    check_max_x = None

    if(isinstance(k, list) and isinstance(std, list)):
        for kk, cconst in zip(k, std):
            new_min_y, new_max_y = plot_chi_square(f=f, ax=ax, start=start, stop=stop, k=kk, std=cconst, step=step)
            check_min_y, check_max_y = select_min_max(check_min_y, check_max_y, new_min_y, new_max_y)
    elif(isinstance(k, list)):
        for kk in k:
            new_min_y, new_max_y = plot_chi_square(f=f, ax=ax, start=start, stop=stop, k=kk, std=std, step=step)
            check_min_y, check_max_y = select_min_max(check_min_y, check_max_y, new_min_y, new_max_y)
    elif(isinstance(std, list)):
        for cconst in std:
            new_min_y, new_max_y = plot_chi_square(f=f, ax=ax, start=start, stop=stop, k=k, std=cconst, step=step)
            check_min_y, check_max_y = select_min_max(check_min_y, check_max_y, new_min_y, new_max_y)
    else:
        new_min_y, new_max_y = plot_chi_square(f=f, ax=ax, start=start, stop=stop, k=k, std=std, step=step)
        check_min_y, check_max_y = select_min_max(check_min_y, check_max_y, new_min_y, new_max_y)

    if(max_y is not None):
        check_max_y = max_y
    if(min_y is not None):
        check_min_y = min_y

    if(max_x is not None):
        check_max_x = max_x
    if(min_x is not None):
        check_min_x = min_x

    return check_min_y, check_max_y, check_min_x, check_max_x

def plot(f, name, **kwargs):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    min_y, max_y, min_x, max_x = plot_chi_square_loop(f=f, ax=ax, **kwargs)
    ax.set_ylim(bottom=min_y, top=max_y)
    ax.set_xlim(left=min_x, right=max_x)
    ax.set_xlabel("z")
    ax.set_ylabel("f(z)")

    ax.legend(title='Parametry')
    ax.set_title('Logarytm funkcji rozkładu chi-kwadrat.')
    fig.savefig(f"tmp/{name}")


plot(f=loss, stop=5, name='chi_square_chart_1.png', k=3, std=[0.5, 0.2], max_y=12, max_x = 10/2, min_x = -1 / 5)
plot(f=loss, stop=5, name='chi_square_chart_2.png', k=10, std=[0.5, 0.2], max_y=12, max_x = 10/2, min_x = -1 / 5)
plot(f=loss, stop=500, name='chi_square_chart_3.png', k=3, std=[5, 2], max_y=12, max_x = 1000/2, min_x = -100 / 5)
plot(f=loss, stop=500, name='chi_square_chart_4.png', k=10, std=[5, 2], max_y=12, max_x = 1000/2, min_x = -100 / 5)
plot(f=loss, stop=1e+5, name='chi_square_chart_5.png', k=3, std=[50, 20], max_y=12, max_x = 1e+5/2, min_x = -1e+4 / 5)
plot(f=loss, stop=1e+5, name='chi_square_chart_6.png', k=10, std=[50, 20], max_y=12, max_x = 1e+5/2, min_x = -1e+4 / 5)
plot(f=loss, stop=1e+4, name='chi_square_chart_7.png', k=3, std=[50, 20], max_y=12, max_x = 1e+4, min_x = -1e+3 / 5)
plot(f=loss, stop=1e+4, name='chi_square_chart_8.png', k=92, std=[5, 2], max_y=12, max_x = 1e+4, min_x = -1e+2/5)
plot(f=loss_k_eq_2, stop=10, start=-10, name='chi_square_chart_9.png', k=2, std=[5, 2], max_y=12)
#plot_chi_square(f=loss_2, stop=40, name='chi_square_loss_chart.png', k=3, std=1)