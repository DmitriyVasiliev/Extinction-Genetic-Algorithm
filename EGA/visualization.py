import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import PyQt5.QtWidgets

import random
import numpy as np


def ret_generation_data():
    data = [random.random() for i in range(25)]
    return data, data


class App(QMainWindow):

    def __init__(self, dataset):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 1200
        self.height = 900
        self.data = dataset
        self.cur_ind = 0
        self.plot = None
        self.initUI()

    def change_cur_ind(self, change):
        self.cur_ind += change
        if self.cur_ind < 0:
            self.cur_ind = 0
        if self.cur_ind >= self.data.__len__():
            self.cur_ind = self.data.__len__() - 1

    def btn_prev(self):
        self.change_cur_ind(-1)
        n = self.cur_ind
        data = self.data[n]
        self.plot.draw_new_data(data, n)

    def btn_next(self):
        self.change_cur_ind(1)
        n = self.cur_ind
        data = self.data[n]
        self.plot.draw_new_data(data, n)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        m = PlotCanvas(self, width=9, height=6)
        m.move(0, 0)
        self.plot = m
        self.btn_prev()

        button = QPushButton('prev', self)
        button.setToolTip('Previous generation')
        button.move(100, 700)
        button.resize(140, 100)
        button.clicked.connect(self.btn_prev)

        button1 = QPushButton('next', self)
        button1.setToolTip('Next generation')
        button1.move(250, 700)
        button1.resize(140, 100)
        button1.clicked.connect(self.btn_next)

        # self.show()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = figure.add_subplot(111)

        FigureCanvas.__init__(self, figure)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # self.plot()
        self.draw()

        # ax = self.figure.add_subplot(111)
        # ax = self.axes
        # fig, ax = plt.subplots()
        # ax.cla()
        # plt.show()
        # self.plot()

    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

    def draw_new_data(self, data, number, var_x=0, var_y=1):
        ax = self.axes
        ax.cla()
        count = data.__len__()
        x1 = np.zeros(shape=(count))
        y1 = np.zeros(shape=(count))
        # x1, y1 = ret_generation_data()
        for x in range(count):
            ind = data[x]
            x1[x] = ind.variables[var_x]
            y1[x] = ind.variables[var_y]
        color = 'green'
        gen_lab = 'Generation ' + number.__str__()
        ax.scatter(x1, y1, c=color, edgecolors='none', s=20, label=gen_lab, alpha=0.8)
        ax.set_title(gen_lab)
        ax.grid(linestyle='--')
        self.draw()
        '''
        x1, y1 = ret_generation_data()
        color = 'green'
        gen_lab = 'Generation 0'
        ax.scatter(x1, y1, c=color, edgecolors='none', s=20, label=gen_lab, alpha=0.8)
        title = 'FOREL для тестов '
        plt.title(title)
        # plt.legend(loc=2)
        plt.grid(linestyle='--')
        self.draw()
        '''


def vis_main(dataset):
    print('Start visualization')
    app = QApplication([])
    # app = QApplication(sys.argv)
    ex = App(dataset)
    ex.show()
    return ex
