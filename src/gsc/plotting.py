"""
Plotting functions

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_TP(vec1, vec2, figsize=None):
    '''Compute the outer product of two vectors and present it in a diagram.'''

    nrow = vec1.shape[0]
    ncol = vec2.shape[0]
    radius = 0.4

    arr = np.zeros((nrow + 1, ncol + 1))
    arr[1:, 1:] = np.outer(vec1, vec2)
    arr[0, 1:] = vec2
    arr[1:, 0] = vec1

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    for ii in range(nrow + 1):
        for jj in range(ncol + 1):
            if (ii == 0) and (jj == 0):
                continue
            if (ii == 0) or (jj == 0):
                alpha = 1  # 0.3
            else:
                alpha = 1

            if arr[ii, jj] >= 0:
                curr_unit = plt.Circle(
                    (jj, -ii), radius,
                    color=plt.cm.gray(1 - abs(arr[ii, jj])),
                    alpha=alpha)
                ax.add_artist(curr_unit)
                curr_unit = plt.Circle(
                    (jj, -ii), radius,
                    color='k', fill=False)
                ax.add_artist(curr_unit)
            else:
                curr_unit = plt.Circle(
                    (jj, -ii), radius,
                    color='k', fill=False)
                ax.add_artist(curr_unit)
                curr_unit = plt.Circle(
                    (jj, -ii), radius - 0.1,
                    color=plt.cm.gray(1 - abs(arr[ii, jj])),
                    alpha=alpha)
                ax.add_artist(curr_unit)
                curr_unit = plt.Circle(
                    (jj, -ii), radius - 0.1,
                    color='k', fill=False)
                ax.add_artist(curr_unit)

    ax.axis([
        0 - radius - 0.6, ncol + radius + 0.6,
        - nrow - radius - 0.6, 0 + radius + 0.6])
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def plot_state(act=None, actC=None, net=N, coord='C',
               colorbar=True, disp=True, grayscale=True):
    """Plot the activation state (conceptual coordinate) in a heatmap."""

    if coord == 'C':
        heatmap(actC.reshape((N.grammar.nF, N.grammar.nR)), xticklabels=net.role2index,
                yticklabels=net.filler2index, grayscale=grayscale,
                colorbar=colorbar, disp=disp, val_range=[0, 1])
    elif coord == 'N':
        act_mat = act.reshape((net.grammar.nF, net.grammar.nR), order='F')
        yticklabels = ['f' + str(ii) for ii in range(net.grammar.nF)]
        xticklabels = ['r' + str(ii) for ii in range(net.grammar.nR)]
        heatmap(
            act_mat, xticklabels=xticklabels, yticklabels=yticklabels,
            grayscale=grayscale, colorbar=colorbar, disp=disp)


def heatmap(data, xlabel=None, ylabel=None, xticklabels=None, yticklabels=None,
            grayscale=False, colorbar=True, rotate_xticklabels=False,
            xtick=True, ytick=True, disp=True, val_range=None):

    # Plot the activation trace as heatmap
    if grayscale:
        cmap = plt.cm.get_cmap("gray_r")
    else:
        cmap = plt.cm.get_cmap("Reds")  # TODO: Find a nicer one

    if val_range is not None:
        plt.imshow(data, cmap=cmap, vmin=val_range[0], vmax=val_range[1],
                   interpolation="nearest", aspect='auto')
    else:
        plt.imshow(data, cmap=cmap, interpolation="nearest", aspect='auto')

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=16)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)
    if xticklabels is not None:
        if rotate_xticklabels:
            plt.xticks(
                np.arange(len(xticklabels)), xticklabels,
                rotation='vertical')
        else:
            plt.xticks(np.arange(len(xticklabels)), xticklabels)

    if yticklabels is not None:
        plt.yticks(np.arange(len(yticklabels)), yticklabels)

    if xtick is False:
        plt.tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom='off',       # ticks along the bottom edge are off
            top='off',          # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
    if ytick is False:
        plt.tick_params(
            axis='y',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            left='off',         # ticks along the bottom edge are off
            right='off',        # ticks along the top edge are off
            labelleft='off')    # labels along the bottom edge are off

    if colorbar:
        plt.colorbar()

    if disp:
        plt.show()


def plot_trace(self, varname):
    """Plot the trace of a given variable"""

    x = self.traces['t']
    if varname == 'actC':
        y = self.N2C(self.traces[varname[:-1]].T).T
    else:
        y = self.traces[varname]

    plt.plot(x, y)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel(varname, fontsize=16)
    plt.grid(True)
    plt.show()


plot_state(act=N.state, actC=N.stateC, net=N)
plt.show()
