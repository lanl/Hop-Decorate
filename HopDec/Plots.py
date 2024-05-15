import os
import math

import scipy
from scipy.interpolate import interp1d
import pylab

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('_mpl-gallery-nogrid')

def linePlot(filename, xvals, xlabel, yvals, ylabel, title=None, label=None, grid=1, textsize=20, useInterp=1, yerror=None,
             xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, ymax=None, tight=True, cmap = None):
    """
    Do a line plot of data using stineman interpolation.
    'ymax' should be specified as the index in the arrays of the maximum
                (so that interpolation doesn't change it).
    'label' goes in the legend if you specify it.
    Saves graph in 'filename'.png.
    
    """
    
    pars = matplotlib.figure.SubplotParams(left=0.14, bottom=0.12, right=None, top=None, wspace=None, hspace=None)
    fig = plt.figure(figsize=(8,6),subplotpars=pars)
    ax = fig.add_subplot(111)

    plt.plot(xvals, yvals, linewidth = 2, color = 'red')
    plt.scatter(xvals, yvals, s=100)

    plt.xlabel(xlabel, fontsize=textsize)
    plt.ylabel(ylabel, fontsize=textsize)
    if title is not None:
        plt.title(title)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(textsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(textsize)
    
    if label is not None:
        leg = plt.legend(loc='best')
        for t in leg.get_texts():
            t.set_fontsize(textsize)
    
    if xlimmin is not None:
        plt.xlim(xmin=xlimmin)
    
    if xlimmax is not None:
        plt.xlim(xmax=xlimmax)
    
    if ylimmin is not None:
        plt.ylim(ymin=ylimmin)
    
    if ylimmax is not None:
        plt.ylim(ymax=ylimmax)
    
    if grid:
        plt.grid()
    
    if tight:
        plt.tight_layout()
    
    plt.savefig(f"{filename}")
    
    plt.clf()
    plt.close()

def histPlot(data,filename = 'test', title=None, label=None, grid=1, textsize=20, useInterp=1, yerror=None,
             xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None, ymax=None, tight=True, cmap = None):
    
    pars = matplotlib.figure.SubplotParams(left=0.14, bottom=0.12, right=None, top=None, wspace=None, hspace=None)
    fig = plt.figure(figsize=(8,6),subplotpars=pars)
    ax = fig.add_subplot(111)

    ax.hist(data)

    if tight:
        plt.tight_layout()
    plt.savefig(f"{filename}.png")

if __name__=="__main__":
    pass

