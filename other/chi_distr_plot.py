import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.distributions.chi2 import Chi2
import numpy as np
from scipy.stats import chi, chi2

#x = np.arange(0, 4, 0.001)

#fig, ax = plt.subplots( nrows=1, ncols=1 )
#plot Chi-square distribution with 4 degrees of freedom
#ax.plot(x, chi2.pdf(x, df=4))
#chi2 = chi


#ax.plot(x, chi2.pdf(x, 1), label='df: 1')
#ax.plot(x, chi2.pdf(x, 4), label='df: 4')
#ax.plot(x, chi2.pdf(x, 8), label='df: 8') 
#ax.plot(x, chi2.pdf(x, 12), label='df: 12') 

#ax.legend(title='Parametry')
#add legend to plot
#fig.savefig('tmp/chi_chart.png')

def plot_chi(df):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    for d in df:
        #stop = chi.ppf(0.99, d)
        stop = 8
        x = np.linspace(chi.ppf(0.01, d), stop, 100)
        ax.plot(x, chi.pdf(x, d), label=f'k={d}')
    ax.legend(title='Parametry')
    ax.set_title('Gęstość prawdopodobieństwa chi')
    fig.savefig('tmp/chi_chart.png')

def plot_chi_square(df):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.set_ylim([0, 0.5])
    for d in df:
        #stop = chi.ppf(0.99, d)
        stop = 8
        x = np.linspace(chi2.ppf(0.01, d), stop, 100)
        ax.plot(x, chi2.pdf(x, d), label=f'k={d}')
    ax.legend(title='Parametry')
    ax.set_title('Gęstość prawdopodobieństwa kwadratu chi')
    fig.savefig('tmp/chi_square_chart.png')

plot_chi([1, 2, 4, 8])
plot_chi_square([1, 2, 4, 8])

