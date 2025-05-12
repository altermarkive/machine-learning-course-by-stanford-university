#!/usr/bin/env python3

import numpy as np
import matplotlib
# Force matplotlib to not use any X Windows backend (must be called befor importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def displayData(X, example_width=None):
    #DISPLAYDATA Display 2D data in a nice grid
    #   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the 
    #   displayed array if requested.

    figsize = (10, 10)

    # Set example_width automatically if not passed in
    example_width = example_width or int(np.round(np.sqrt(X.shape[1])))

    # Gray Image
    cmap='Greys'

    # Compute rows, cols
    m, n = X.shape
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 0.025

    # Display Image
    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=pad, hspace=pad)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'), cmap=cmap, extent=[0, 1, 0, 1])
        # Do not show axis
        ax.axis('off')

    return fig, ax_array
    #end
