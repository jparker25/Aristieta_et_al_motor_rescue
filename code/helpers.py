"""
helpers.py

Various helper functions for fixing plots and running terminal commands.

Author: John E. Parker (2024)
"""

# python modules
import os, string
import numpy as np


def run_cmd(str, print_out=True):
    """
    Runs a given string on terminal.

    Parameters
    -------
    \t str (string) : Command to be passed to terminal

    \t print_out=True (boolean) : Boolean to print command.
    """
    if print_out:
        print()
        print(str)
    os.system(str)
    if print_out:
        print()


def makeNice(axes, labelsize=8, lw=0.5, width=0.5):
    """
    Edits given axes/list of axes to set desired layout.

    Parameters
    -------
    \t axes (list) : List of axes to be edited.

    \t labelsize=8 (int) : Size of x and y ticks.

    \t lw=0.5 (double) : Thickness of spines for axes.

    \t width=0.5 (double) : Width of ticks on spines.
    """

    # Iterate through list of axes
    if type(axes) == list:
        for ax in axes:
            # Iterate through all spines
            for i in ["left", "right", "top", "bottom"]:
                # Remove right and top spines
                if i != "left" and i != "bottom":
                    ax.spines[i].set_visible(False)
                    ax.tick_params("both", width=0, labelsize=labelsize)
                # Adjust left and bottom spines by ticksizes and tick labels
                else:
                    ax.spines[i].set_linewidth(lw)
                    ax.tick_params("both", width=width, labelsize=labelsize)
    # Adjust one axis
    else:
        # Iterate through all spines
        for i in ["left", "right", "top", "bottom"]:
            # Remove right and top spines
            if i != "left" and i != "bottom":
                axes.spines[i].set_visible(False)
                axes.tick_params("both", width=0, labelsize=labelsize)
            # Adjust left and bottom spines by ticksizes and tick labels
            else:
                axes.spines[i].set_linewidth(lw)
                axes.tick_params("both", width=width, labelsize=labelsize)


def add_fig_labels(axes, fontsize=10):
    """
    Adds subplot labels (ascii lowercase) to a list of axes.

    Parameters
    -------
    \t axes (list) : List of axes to be labeled (ascii lowercase).

    \t labelsize=10 (int) : Font size of subplot label
    """
    labels = string.ascii_lowercase
    for i in range(len(axes)):
        axes[i].text(
            -0.15,
            1.05,
            labels[i],
            fontsize=fontsize,
            transform=axes[i].transAxes,
            color="black",
        )


def match_axis(axes, type="both"):
    """
    Matches limits in x, y, or both axis for list of plots.

    Parameters
    -------
    \t axes (list) : List of axes to be edited.

    \t type="both" (string) : Designate which axes to match, "x", "y", or "both"
    """
    if type == "x":
        min = np.min([ax.get_xlim()[0] for ax in axes])
        max = np.max([ax.get_xlim()[1] for ax in axes])
        for ax in axes:
            ax.set_xlim([min, max])
    elif type == "y":
        min = np.min([ax.get_ylim()[0] for ax in axes])
        max = np.max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_ylim([min, max])
    else:
        match_axis(axes, type="x")
        match_axis(axes, type="y")
