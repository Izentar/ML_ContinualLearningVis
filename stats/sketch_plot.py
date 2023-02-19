import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def get_x(mini, maxi, density):
        diff = np.abs(maxi - mini)
        x = np.linspace(mini, maxi, density * diff)
        return x

def sketch_plot(f, mini, maxi, density, color='red'):
    x = get_x(mini, maxi, density)
    #x = np.array(range(-5, 50), dtype=np.float32)
    plt.plot(x, f(x), color=color)
    plt.grid()
'''
def model_inverse(x, sigma, eps, k):
    dims = (k / 2 - 1)
    z = stats.chi2.pdf(x, df=k)
    z = np.array(z)
    z /= 2 * sigma**2

    z += eps

    return z
'''
def model_chi(x, sigma, eps, k):
    z = np.array(x)
    #z = np.array(stats.chi2.pdf(x, df=k))
    z /= 2 * sigma**2
    log = np.log((z / k) + eps)
    return -((k / 2 - 1) * log) + (z / (2 * k)) 
'''
def model_chi2(x, sigma, eps, k):
    z = stats.chi2.pdf(x, df=k)
    log = np.log(z / 2*k / sigma**2)
    return -((k / 2 - 1) * log) + (0.5 * z / 2*k/sigma**2) 
'''

def chi2(x, k):
    return stats.chi2.pdf(x, df=k)
'''
def chi2_sample(k):
    return np.random.chisquare(k, k)

def plot_batch_points_sum(f, batch_size, class_prob):
    batch = []
    for b in range(batch_size):
        y_points = f()
        t = np.random.uniform(0, 1)
        if(t > class_prob):
            batch.append(-1 * y_points)
        else:
            batch.append(y_points)

    
    point = numpy.sum(batch, axis=0)

    plt.plot(x, batch, color='red')
    plt.grid()
    plt.show()

def plot_batch(f, x, batch_size, class_prob):
    batch = np.zeros_like(x)
    for b in range(batch_size):
        y = f(x)
        t = np.random.uniform(0, 1)
        if(t > class_prob):
            batch = np.add(batch, -1 * y)
        else:
            batch = np.add(batch, y)
    
    plt.plot(x, batch, color='red')
    plt.grid()
    plt.show()
'''
def run():
    sigma = 0.5
    eps = 1e-5
    k = 10
    mini = -10
    maxi = 45
    f = lambda x: model_chi(x=x, sigma=0.5, eps=eps, k=k)
    f2 = lambda x: model_chi(x, 0.2, eps, k)
    #f = lambda x: chi2(x, k)

    sketch_plot(f, mini, maxi, 1)
    sketch_plot(f2, mini, maxi, 10, 'g')
    plt.grid()
    plt.show()

'''
def run2():
    sigma = 0.2
    eps = 0.
    k = 32

    x = get_x(-10, 300, 10)
    f = lambda x: model_chi(x, sigma, eps, k)
    plot_batch(f, x, k, 0.55)
'''
if __name__ == '__main__':
    run()