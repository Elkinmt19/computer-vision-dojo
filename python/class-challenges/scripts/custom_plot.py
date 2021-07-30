# External imports
import matplotlib.pylab as plt
import numpy as np


def create_plots(x_array, y_array, tittle, x_label, y_label):
        """
        Create simple x-y-z position and x-y-z orientatio plots based on the 
        simultations done in the "execute_simulation" method.
        It returns two pop-out matplotlib graphs.
        """

        # Define a figure for the creation of the plot
        figure_1 = plt.figure(1)
        axes = figure_1.add_axes([0.1, 0.1, 0.8, 0.8])

        # Generate the plot to the figure_1 and its axes.
        axes.plot(x_array, y_array, c='b', linewidth=2)

        # Customize figure_1 with title, "x"-"y" labels
        axes.set_title(tittle)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)

        # Change the background color of the external part
        figure_1.patch.set_facecolor((0.2, 1, 1))

        # Display the created figures of this script
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()