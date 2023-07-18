import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

def loss(value, k, const, eps=1e-6):
    first_part= -(k / 2 - 1) * np.log(value / (2*k) / const**2 + eps)
    second_part = 0.5 * value / (2 * k) / const**2
    return first_part + second_part

def loss_2(value, k, const, eps=1e-6):
    value = value / (2 * const**2) 
    first_part= -(k / 2 - 1) * np.log(value / k + eps)
    second_part = value / (2 * k)
    return first_part + second_part

def select_min_max(minp, maxp, new_minp, new_maxp):
    if(minp < new_minp):
        new_minp = minp
    if(maxp > new_maxp):
        new_maxp = maxp
    return new_minp, new_maxp

def plot_chi_square(f, ax, stop, k, const):
    x = np.linspace(0, stop, 100)
    output = list(map(lambda x: f(value=x, const=const, k=k), x))
    ax.plot(x, output, label=f'chi loss: const={const}; k={k}')
    return min(output), max(output)

def plot_chi_square_loop(f, ax, stop, k, const):
    minp = 0
    maxp = 0
    if(isinstance(k, list) and isinstance(const, list)):
        for kk, cconst in zip(k, const):
            new_minp, new_maxp = plot_chi_square(f=f, ax=ax, stop=stop, k=kk, const=cconst)
            minp, maxp = select_min_max(minp, maxp, new_minp, new_maxp)
    elif(isinstance(k, list)):
        for kk in k:
            new_minp, new_maxp = plot_chi_square(f=f, ax=ax, stop=stop, k=kk, const=const)
            minp, maxp = select_min_max(minp, maxp, new_minp, new_maxp)
    elif(isinstance(const, list)):
        for cconst in const:
            new_minp, new_maxp = plot_chi_square(f=f, ax=ax, stop=stop, k=k, const=cconst)
            minp, maxp = select_min_max(minp, maxp, new_minp, new_maxp)
    else:
        new_minp, new_maxp = plot_chi_square(f=f, ax=ax, stop=stop, k=k, const=const)
        minp, maxp = select_min_max(minp, maxp, new_minp, new_maxp)
    
    return minp, maxp

def plot(f, name, **kwargs):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    minp, maxp = plot_chi_square_loop(f=f, ax=ax, **kwargs)
    ax.set_ylim([minp-1, maxp+1])

    ax.legend(title='Parametry')
    ax.set_title('Funkcja używana na wartościach z rozkładu chi-kwadrat.')
    fig.savefig(f"tmp/{name}")

plot(f=loss, stop=100, name='chi_square_chart.png', k=10, const=[1, 0.4])
plot(f=loss, stop=100, name='chi_square_chart_2.png', k=10, const=[0.5, 0.2])
plot(f=loss, stop=1e+6, name='chi_square_chart_3.png', k=10, const=[50, 20])
#plot_chi_square(f=loss_2, stop=40, name='chi_square_loss_chart.png', k=3, const=1)