import numpy as np
import xlwt
import datetime
from PyQt5 import QtWidgets, uic
import sys
import xlrd
import utest
from ranking import Ranking, FRACTIONAL
import os


class App(QtWidgets.QMainWindow, utest.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.bGo.clicked.connect(self.init_Utest)
        self.bGo.clicked.connect(self.folder_test)
        # self.var_ar = ['v10', 'v20', 'v30']
        self.var_ar = ['v2', 'v5', 'v10']
        # self.var_ar = ['v5', 'v10', 'v20']

    def array_indexes(self, test_name):
        sp = test_name.split('_')
        i = 0
        j = 0
        if sp[0] == 'ackley':
            i = 0
        elif sp[0] == 'cigar':
            i = 1
        elif sp[0] == 'griewank':
            i = 2
        elif sp[0] == 'rastrigin':
            i = 3
        elif sp[0] == 'rosenbrock':
            i = 4
        elif sp[0] == 'schaffer':
            i = 5
        elif sp[0] == 'schwefel':
            i = 6
        elif sp[0] == 'sphere':
            i = 7
        else:
            raise Exception('FUNCTION NOT FOUND')

        if sp[1] == self.var_ar[0]:
            j = 0
        elif sp[1] == self.var_ar[1]:
            j = 1
        elif sp[1] == self.var_ar[2]:
            j = 2
        else:
            raise Exception('FUNCTION NOT FOUND')

        return i, j

    def folder_test(self):
        folder1 = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder 1 (Base)",
                                                             "/home/tests")
        print(folder1)
        l_o_f = os.listdir(folder1)
        fold1_list = []
        for file in l_o_f:
            lastn = file[-4:]
            if lastn == '.xls':
                fold1_list.append(file)
        print('.xls in folder: ' + str(fold1_list.__len__()))
        folder2 = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder 2",
                                                             "/home/tests")
        print(folder2)
        l_o_f = os.listdir(folder2)
        fold2_list = []
        for file in l_o_f:
            lastn = file[-4:]
            if lastn == '.xls':
                fold2_list.append(file)
        print('.xls in folder: ' + str(fold2_list.__len__()))
        minl = min(fold1_list.__len__(), fold2_list.__len__())

        test_list = []
        res_ar = np.zeros(dtype=np.int, shape=(8, 3))
        val_ar = np.zeros(shape=(24, 10))  # x: func_vars, y: min1, med1, mean1, std1, max1, 2:...
        for file in fold1_list:
            ind = file.find('_')
            s = file[(ind + 1):]
            s = s[:-4]
            test_list.append(s)
        print(test_list)
        for i in range(test_list.__len__()):
            name = test_list[i]
            t1 = None
            t2 = None
            for x in fold1_list:
                if x.find(name) != -1:
                    t1 = x
                    break
            for x in fold2_list:
                if x.find(name) != -1:
                    t2 = x
                    break
            ar1 = self.xl_to_array(folder1 + '/' + t1)
            ar2 = self.xl_to_array(folder2 + '/' + t2)
            res = MannWhitneyU(ar1, ar2)
            j, k = self.array_indexes(name)
            res_ar[j, k] = res
            xx = j * 3 + k
            s1 = np.sort(ar1)
            s2 = np.sort(ar2)
            n1 = ar1.shape[0]
            n2 = ar2.shape[0]
            min1 = s1[0]
            min2 = s2[0]
            med1 = s1[int(n1 / 2)]
            med2 = s2[int(n2 / 2)]
            max1 = s1[n1 - 1]
            max2 = s2[n2 - 1]
            val_ar[xx, 0] = min1
            val_ar[xx, 1] = med1
            val_ar[xx, 2] = np.mean(s1)
            val_ar[xx, 3] = np.std(s1)
            val_ar[xx, 4] = max1
            val_ar[xx, 5] = min2
            val_ar[xx, 6] = med2
            val_ar[xx, 7] = np.mean(s2)
            val_ar[xx, 8] = np.std(s2)
            val_ar[xx, 9] = max2
        # record

        book = xlwt.Workbook(encoding="utf-8")
        sheet = book.add_sheet("Test result")
        # styling
        f = xlwt.Font()
        s = xlwt.XFStyle()
        f.name = 'Times New Roman'
        f.charset = 10
        s.font = f
        f2 = xlwt.Font()
        f2.name = 'Times New Roman'
        f2.charset = 10
        f2.bold = True
        sb = xlwt.XFStyle()
        sb.font = f2

        sheet.col(0).width = 3000
        sheet.col(1).width = 3200
        sheet.col(2).width = 3200

        sheet.col(5).width = 3600
        sheet.col(6).width = 2500
        sheet.col(7).width = 2500
        sheet.col(8).width = 2500
        sheet.col(9).width = 2500
        sheet.col(10).width = 2500
        sheet.col(11).width = 2500
        sheet.col(12).width = 2500
        sheet.col(13).width = 2500
        sheet.col(14).width = 2500
        sheet.col(15).width = 2500
        sheet.col(16).width = 2500
        sheet.col(17).width = 2500

        sheet.write(0, 0, 'Base folder', s)
        sheet.write(0, 1, folder1, s)
        sheet.write(1, 0, 'Second folder', s)
        sheet.write(1, 1, folder2, s)
        sheet.write(2, 0, '0 means equal', s)
        sheet.write(2, 1, '1 Base is better', s)
        sheet.write(2, 2, '-1 Second is better', s)
        sheet.write(3, 1, self.var_ar[0], s)
        sheet.write(3, 2, self.var_ar[1], s)
        sheet.write(3, 3, self.var_ar[2], s)
        for i in range(8):
            for j in range(3):
                val = res_ar[i, j]
                sheet.write(i + 4, j + 1, str(val), s)
        sheet.write(4, 0, 'ackley', s)
        sheet.write(5, 0, 'cigar', s)
        sheet.write(6, 0, 'griewank', s)
        sheet.write(7, 0, 'rastrigin', s)
        sheet.write(8, 0, 'rosenbrock', s)
        sheet.write(9, 0, 'schaffer', s)
        sheet.write(10, 0, 'schwefel', s)
        sheet.write(11, 0, 'sphere', s)
        space = 6
        sheet.write(2, 0 + space, 'Base', s)
        sheet.write(2, 0 + space + 5, 'Second', s)
        sheet.write(3, 0 + space, 'Min', s)
        sheet.write(3, 0 + space + 1, 'Med', s)
        sheet.write(3, 0 + space + 2, 'Mean', s)
        sheet.write(3, 0 + space + 3, 'STD', s)
        sheet.write(3, 0 + space + 4, 'Max', s)
        sheet.write(3, 0 + space + 5, 'Min', s)
        sheet.write(3, 0 + space + 6, 'Med', s)
        sheet.write(3, 0 + space + 7, 'Mean', s)
        sheet.write(3, 0 + space + 8, 'STD', s)
        sheet.write(3, 0 + space + 9, 'Max', s)
        for i in range(24):
            name = test_list[i]
            j, k = self.array_indexes(name)
            xx = j * 3 + k
            sheet.write(4 + xx, space - 1, name, s)
            t_ar = np.zeros(dtype=np.bool, shape=(10))
            for j in range(5):
                jj = j + 5
                if val_ar[i, j] < val_ar[i, jj]:
                    t_ar[j] = True
                    t_ar[jj] = False
                else:
                    t_ar[jj] = True
                    t_ar[j] = False

            for j in range(10):
                val = val_ar[i, j]
                # val = round(val, 8)
                val = np.format_float_scientific(val, precision=3, trim='0')
                if t_ar[j]:
                    sheet.write(4 + i, space + j, str(val), sb)
                else:
                    sheet.write(4 + i, space + j, str(val), s)
        book.save("output.xls")
        print('END')

    def init_Utest(self):
        # базовый, второй изменённый
        fileName1 = QtWidgets.QFileDialog.getOpenFileName(self, "XLS 1 (Base)",
                                                          "/home/tests", "XLS files (*.xls)")

        ar1 = self.xl_to_array(fileName1[0])
        fileName2 = QtWidgets.QFileDialog.getOpenFileName(self, "XLS 2",
                                                          "/home/tests", "XLS files (*.xls)")
        ar2 = self.xl_to_array(fileName2[0])
        res = MannWhitneyU(ar1, ar2)
        res_t = 'First file = ' + str(fileName1) + '\n'
        res_t += 'Second file = ' + str(fileName2) + '\n'
        if res == 0:
            res_t += 'Testes are equal'
        elif res == -1:
            res_t += 'Second test is better, first - worse'
        elif res == 1:
            res_t += 'First test is better, second - worse'
        else:
            res_t += 'ERROR'

        s1 = np.sort(ar1)
        s2 = np.sort(ar2)
        n1 = ar1.shape[0]
        n2 = ar2.shape[0]
        min1 = s1[0]
        min2 = s2[0]
        med1 = s1[int(n1 / 2)]
        med2 = s2[int(n2 / 2)]
        max1 = s1[n1 - 1]
        max2 = s2[n2 - 1]
        mean1 = np.mean(s1)
        mean2 = np.mean(s2)
        std1 = np.std(s1)
        std2 = np.std(s2)

        res_t += '\n'
        res_t += 'First test: min = ' + str(min1) + '; med = ' + str(med1) + '; max = ' + str(max1) + '; mean = ' + str(
            mean1) + '; std = ' + str(std1)
        res_t += '\n'
        res_t += 'Second test: min = ' + str(min2) + '; med = ' + str(med2) + '; max = ' + str(
            max2) + '; mean = ' + str(
            mean2) + '; std = ' + str(std2)
        mb = QtWidgets.QMessageBox()
        print(res_t)
        mb.setWindowTitle('Result')
        mb.setText(res_t)
        mb.exec_()

        '''
        if ar1 is None or ar2 is None:
            print('Error. End of program')
        rank_ar = list()
        for x in ar1:
            a = ('1', x)
            rank_ar.append(a)
        print('End')
        '''
        # Get values from 2 sets
        # Sort values
        # Set corresponded rang
        # Calc N1, N2 (30, 30)
        # U crit for 30, 30 = 317
        # Calc T1, T2 (sums of ranks)
        # Calc U1, U2 (=T-(N*(N+1))/2)
        # NMax, MaxT
        # Calc U (=(N1*N2)+((NMax*(NMax+1))/2)-MaxT)
        # if U > U_crit => same, else different

    def xl_to_array(self, path):

        # get the first worksheet
        first_sheet = None
        book = xlrd.open_workbook(path)
        try:
            first_sheet = book.sheet_by_name('N Runs')
            # first_sheet = book1.sheet_by_index(0)
        except:
            print('Couldn\'t find sheet \'N Runs\'')
            return None

        l_ind = first_sheet.nrows - 1
        '''
        c_cell = first_sheet.cell(l_ind + 1, 0).value
        while c_cell != '':
            l_ind += 1
            try:
                c_cell = first_sheet.cell(l_ind + 1, 0).value
            except:
                l_ind -= 1
                break
        '''
        ar = np.zeros(shape=(l_ind))

        for i in range(l_ind):
            cell = first_sheet.cell(i + 1, 0)
            cell = cell.value
            if cell != '':
                try:
                    cell = str(cell)
                    cell = cell.replace(',', '.')
                    a = float(cell)
                    ar[i] = a
                except:
                    print('Values not valid!')
                    break
        return ar

    '''
            # read a row
        print(first_sheet.row_values(0))

        # read a cell

        print(cell.value)

        # read a row slice
        print(first_sheet.row_slice(rowx=0,
                                    start_colx=0,
                                    end_colx=2))

        print(book1.nsheets)
    '''


def MannWhitneyU(Sample1, Sample2):
    NewSample = np.concatenate((Sample1, Sample2), axis=0)
    NewRanks, Groups = get_fract_ranks_and_groups(NewSample)
    SumRanks = 0
    SumRanks2 = 0
    for i in range(Sample1.shape[0]):
        SumRanks += NewRanks[i]
        SumRanks2 += NewRanks[Sample1.shape[0] + i]
    U1 = SumRanks - Sample1.shape[0] * (Sample1.shape[0] + 1.0) / 2.0
    U2 = SumRanks2 - Sample2.shape[0] * (Sample2.shape[0] + 1.0) / 2.0
    Umean = Sample1.shape[0] * Sample2.shape[0] / 2.0
    GroupsSum = 0
    for index in Groups:
        GroupsSum += (index * index * index - index) / 12
    N = Sample1.shape[0] + Sample2.shape[0]
    part1 = Sample1.shape[0] * Sample2.shape[0] / (N * (N - 1.0))
    part2 = (N * N * N - N) / 12.0
    Ucorr2 = np.sqrt(part1 * (part2 - GroupsSum))
    Z1 = (U1 - Umean) / Ucorr2
    Z2 = (U2 - Umean) / Ucorr2
    if (Z1 <= Z2):
        if (Z1 < -2.58):
            # print("worse")
            return 1
    else:
        if (Z2 < -2.58):
            # print("better")
            return -1
    # print("equal")
    return 0


def get_fract_ranks_and_groups(data):
    sort_index = np.argsort(-data)
    sort_list = -np.sort(-data)
    groups = []
    my_new_ranks = np.zeros(data.shape[0])
    counter = 0
    while (True):
        if (counter == data.shape[0]):
            break
        if (counter == data.shape[0] - 1):
            my_new_ranks[counter] = counter
            break
        if (sort_list[counter] != sort_list[counter + 1]):
            my_new_ranks[counter] = counter
            counter += 1
        else:
            avgrank = 0
            start = counter
            while (sort_list[start] == sort_list[counter]):
                avgrank += counter
                counter += 1
                if (counter == data.shape[0]):
                    break
            avgrank = avgrank / (counter - start)
            groups.append(counter - start)
            for i in range(start, counter):
                my_new_ranks[i] = avgrank
    index_rank = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        new_rank_inv = data.shape[0] - my_new_ranks[i]
        index_rank[sort_index[i]] = new_rank_inv
    return index_rank, groups


def start_window():
    app = QtWidgets.QApplication([])
    window = App()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    sys.exit(app.exec())


start_window()
