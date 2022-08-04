import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def get_x(min, max, density):
        diff = np.abs(max - min)
        x = np.linspace(-10, 300, 10 * diff)
        return x

def sketch_plot(f, min, max, density):
    x = get_x(min, max, density)
    
    plt.plot(x, f(x), color='red')
    plt.grid()
    plt.show()

def model_chi(x, sigma, eps, k):
    dims = (k / 2 - 1)
    z = stats.chi2.pdf(x, df=k)
    z /= 2 * sigma**2

    return -(dims * np.log(z / k + eps)) + (z / (2 * k))

def chi2(x, k):
    return stats.chi2.pdf(x, df=k)

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

def run():
    sigma = 0.2
    eps = 1e-5
    k = 32
    min = -10
    max = 800
    f = lambda x: model_chi(x, sigma, eps, k)
    #f = lambda x: chi2(x, k)

    sketch_plot(f, min, max, 10)

def run2():
    sigma = 0.2
    eps = 0.
    k = 32

    x = get_x(-10, 300, 10)
    f = lambda x: model_chi(x, sigma, eps, k)
    plot_batch(f, x, k, 0.55)

if __name__ == '__main__':
    run()