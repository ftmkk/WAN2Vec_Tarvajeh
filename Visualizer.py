import matplotlib.pyplot as plt
from bidi import algorithm as bidialg
import arabic_reshaper


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def get_persian_text(text):
        reshaped_text = arabic_reshaper.reshape(text)
        artext = bidialg.get_display(reshaped_text)
        return artext

    def scatter_plot(self, x_coords, y_coords, labels):
        for i, label in enumerate(labels):
            x = x_coords[i]
            y = y_coords[i]
            plt.scatter(x, y, alpha=0.6)
            plt.text(x + 0.03, y + 0.03, self.get_persian_text(label), name='Times New Roman')
        plt.show();
