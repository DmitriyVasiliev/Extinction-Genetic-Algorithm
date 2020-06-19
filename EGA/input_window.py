# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'input_window.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(692, 515)
        MainWindow.setMinimumSize(QtCore.QSize(692, 515))
        MainWindow.setMaximumSize(QtCore.QSize(692, 515))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 55, 16))
        self.label_2.setObjectName("label_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 30, 661, 391))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setLineWidth(2)
        self.frame.setObjectName("frame")
        self.gridLayoutWidget = QtWidgets.QWidget(self.frame)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 651, 381))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(5, 5, 5, 5)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.spin_calc = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spin_calc.setMinimum(1000)
        self.spin_calc.setMaximum(100000000)
        self.spin_calc.setSingleStep(10000)
        self.spin_calc.setProperty("value", 5000)
        self.spin_calc.setObjectName("spin_calc")
        self.gridLayout.addWidget(self.spin_calc, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.spin_runs = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spin_runs.setMinimum(1)
        self.spin_runs.setMaximum(1000)
        self.spin_runs.setSingleStep(10)
        self.spin_runs.setProperty("value", 100)
        self.spin_runs.setObjectName("spin_runs")
        self.gridLayout.addWidget(self.spin_runs, 2, 1, 1, 1)
        self.cb_func = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.cb_func.setObjectName("cb_func")
        self.gridLayout.addWidget(self.cb_func, 3, 1, 1, 1)
        self.spin_vars = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spin_vars.setMinimum(1)
        self.spin_vars.setMaximum(100)
        self.spin_vars.setProperty("value", 2)
        self.spin_vars.setObjectName("spin_vars")
        self.gridLayout.addWidget(self.spin_vars, 0, 1, 1, 1)
        self.spin_max_pop = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spin_max_pop.setMinimum(10)
        self.spin_max_pop.setMaximum(1000)
        self.spin_max_pop.setSingleStep(10)
        self.spin_max_pop.setProperty("value", 100)
        self.spin_max_pop.setObjectName("spin_max_pop")
        self.gridLayout.addWidget(self.spin_max_pop, 0, 3, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 3, 2, 1, 1)
        self.spin_max_life = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.spin_max_life.setMinimum(0.3)
        self.spin_max_life.setMaximum(1.0)
        self.spin_max_life.setSingleStep(0.01)
        self.spin_max_life.setProperty("value", 0.95)
        self.spin_max_life.setObjectName("spin_max_life")
        self.gridLayout.addWidget(self.spin_max_life, 3, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 2, 1, 1)
        self.spin_mut_p = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.spin_mut_p.setMaximum(1.0)
        self.spin_mut_p.setSingleStep(0.01)
        self.spin_mut_p.setProperty("value", 0.05)
        self.spin_mut_p.setObjectName("spin_mut_p")
        self.gridLayout.addWidget(self.spin_mut_p, 2, 3, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 2, 2, 1, 1)
        self.spin_init_pop = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spin_init_pop.setMinimum(10)
        self.spin_init_pop.setMaximum(1000)
        self.spin_init_pop.setSingleStep(10)
        self.spin_init_pop.setProperty("value", 100)
        self.spin_init_pop.setObjectName("spin_init_pop")
        self.gridLayout.addWidget(self.spin_init_pop, 1, 3, 1, 1)
        self.spin_decay_step = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.spin_decay_step.setMaximum(0.3)
        self.spin_decay_step.setSingleStep(0.01)
        self.spin_decay_step.setProperty("value", 0.05)
        self.spin_decay_step.setObjectName("spin_decay_step")
        self.gridLayout.addWidget(self.spin_decay_step, 4, 3, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 4, 2, 1, 1)
        self.spin_archive_p = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.spin_archive_p.setMaximum(1.0)
        self.spin_archive_p.setSingleStep(0.01)
        self.spin_archive_p.setProperty("value", 0.05)
        self.spin_archive_p.setObjectName("spin_archive_p")
        self.gridLayout.addWidget(self.spin_archive_p, 4, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 4, 0, 1, 1)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 440, 651, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bSRun = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.bSRun.setObjectName("bSRun")
        self.horizontalLayout.addWidget(self.bSRun)
        self.bNRuns = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.bNRuns.setObjectName("bNRuns")
        self.horizontalLayout.addWidget(self.bNRuns)
        self.bSpecTest = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.bSpecTest.setObjectName("bSpecTest")
        self.horizontalLayout.addWidget(self.bSpecTest)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Extinction Evolution v1.0"))
        self.label_2.setText(_translate("MainWindow", "Input"))
        self.label_4.setText(_translate("MainWindow", "Calculations of function"))
        self.label_3.setText(_translate("MainWindow", "Variables Count"))
        self.label.setText(_translate("MainWindow", "Runs"))
        self.label_5.setText(_translate("MainWindow", "Function"))
        self.label_8.setText(_translate("MainWindow", "Init Pop Count"))
        self.label_10.setText(_translate("MainWindow", "Max Life P"))
        self.label_6.setText(_translate("MainWindow", "Max Pop Count"))
        self.label_9.setText(_translate("MainWindow", "Mutation P"))
        self.label_12.setText(_translate("MainWindow", "Decay Step"))
        self.label_11.setText(_translate("MainWindow", "Archive Use P"))
        self.bSRun.setText(_translate("MainWindow", "Single Run"))
        self.bNRuns.setText(_translate("MainWindow", "N Runs"))
        self.bSpecTest.setText(_translate("MainWindow", "Special Test (takes MANY time)"))