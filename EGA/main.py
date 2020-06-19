import matplotlib.pyplot as plt
import numpy as np
import genetic_algoritm as ga
import xlwt
import datetime
import input_window  # Interface
from PyQt5 import QtWidgets, uic
import sys
import visualization

windows = list()


class App(QtWidgets.QMainWindow, input_window.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        lfs = ['sphere', 'cigar', 'rosenbrock', 'ackley', 'griewank', 'rastrigin', 'schaffer', 'schwefel']
        self.cb_func.addItems(lfs)
        self.bSRun.clicked.connect(self.init_s)
        self.bNRuns.clicked.connect(self.init_n)
        self.bSpecTest.clicked.connect(self.init_test)

    def init_test(self):
        test_runs(self.param_dict_init())

    def init_s(self):
        single_run(self.param_dict_init())

    def init_n(self):
        n_runs(self.param_dict_init())

    def param_dict_init(self):
        f_name = self.cb_func.currentText()
        var_count = self.spin_vars.value()
        if f_name == 'cassini2':
            var_count = 22
        bord_all = define_borders(f_name, var_count)
        calc_count = self.spin_calc.value()
        runs_count = self.spin_runs.value()
        max_pop_count = self.spin_max_pop.value()
        init_pop_count = self.spin_init_pop.value()
        mut_p = self.spin_mut_p.value()
        max_lp = self.spin_max_life.value()
        decay_step = self.spin_decay_step.value()
        archive_p = self.spin_archive_p.value()

        # Run!
        param_dict = dict()
        param_dict['function name'] = f_name
        param_dict['is maximization'] = False
        param_dict['variables count'] = var_count
        param_dict['function borders'] = bord_all
        param_dict['calculations count'] = calc_count
        param_dict['runs count'] = runs_count
        param_dict['maximum individuals count'] = max_pop_count
        param_dict['initial individuals count'] = init_pop_count
        param_dict['archive individuals count'] = 1
        param_dict['mutation probability'] = mut_p
        param_dict['maximal life probability'] = max_lp
        param_dict['decay step'] = decay_step
        param_dict['archive resurrect probability'] = archive_p
        param_dict['archive use in crossover probability'] = archive_p
        param_dict['history'] = False
        param_dict['use variety'] = False
        param_dict['sorting balance'] = 1.  # 0. - only distance, 1. - only value
        param_dict['hybrid mutation balance'] = 0.5  # 0. - only D.E., 1. - only polynomial
        param_dict['is DA'] = False
        param_dict['is log'] = False
        return param_dict


def start_window():
    # crossover_test()
    # mutation_test()
    # app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    app = QtWidgets.QApplication([])
    window = App()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    # app.exec_()  # и запускаем приложение
    sys.exit(app.exec())


def define_borders(func_name, var_count):
    bord = [-100., 100.]
    bord_all = np.zeros(shape=(var_count, 2))
    if func_name == 'ackley':
        bord = [-10., 30.]
    elif func_name == 'griewank':
        bord = [-600., 600.]
    elif func_name == 'rastrigin':
        bord = [-5.12, 5.12]
    elif func_name == 'schwefel':
        bord = [-500., 500.]
    for x in range(var_count):
        bord_all[x, :] = bord

    if func_name == 'cassini2':
        bord_all[0] = [-1000, 0]
        bord_all[1] = [3, 5]
        bord_all[2:4] = [0, 1]
        bord_all[4] = [100, 400]
        bord_all[5] = [100, 500]
        bord_all[6] = [30, 300]
        bord_all[7] = [400, 1600]
        bord_all[8] = [800, 2200]
        bord_all[9:14] = [0.01, 0.9]
        bord_all[14:16] = [1.05, 6]
        bord_all[16] = [1.15, 6.5]
        bord_all[17] = [1.7, 291]
        bord_all[18:22] = [-3.141592653589793, 3.141592653589793]
    return bord_all


def single_run(param_dict):
    # vars
    var_count = param_dict['variables count']
    param_dict['history'] = True
    calc_count = param_dict['calculations count']
    f_name = param_dict['function name']

    # Run!

    if f_name == 'cassini2':
        print('matlab init')
        cassini2_problem.process_init()
        print('matlab init end!')

    pop = ga.Population(param_dict)
    pop.evolve()

    if f_name == 'cassini2':
        cassini2_problem.process_term()

    best = pop.archive[0]
    print('Archive best:')
    print(best)
    print('Pop best:')
    print(pop.individuals[0])
    print('Mean lifeteme = ' + str(pop.AV_mean_lifetime))
    print('Archive used: ' + str(pop.AV_archive_use_count) + ' times')
    print('Mutations count: ' + str(pop.AV_mutations_count))
    print('----------end----------')

    # output
    # Initialize a workbook
    book = xlwt.Workbook(encoding="utf-8")
    # Add a sheet to the workbook
    sheet_arch = book.add_sheet("Archive")
    sheet_arch.write(0, 0, 'Values')
    sheet_arch.write(0, 1, 'Variables')
    for i in range(pop.archive.__len__()):
        ind = pop.archive[i]
        sheet_arch.write(i + 1, 0, ind.value.__str__())
        for j in range(var_count):
            sheet_arch.write(i + 1, j + 1, ind.variables[j])

    sheet_pop = book.add_sheet("Population")
    sheet_pop.write(0, 0, 'Values')
    sheet_pop.write(0, 1, 'Variables')
    for i in range(pop.individuals.__len__()):
        ind = pop.individuals[i]
        sheet_pop.write(i + 1, 0, ind.value.__str__())
        for j in range(var_count):
            sheet_pop.write(i + 1, j + 1, ind.variables[j])

    # info
    now = datetime.datetime.now()
    ns = now.strftime("%Y-%m-%d %H:%M:%S")
    s_info = book.add_sheet("Info")
    s_info.write(0, 0, "Date-time")
    s_info.write(0, 1, ns)

    i = 1
    for x in param_dict.keys():
        s_info.write(i, 0, str(x))
        i += 1
    i = 1
    for x in param_dict.values():
        s_info.write(i, 1, str(x))
        i += 1

    book.save("output_single.xls")


    # additional graph
    fig, ax = plt.subplots(figsize=(12, 7))
    # ax.cla()
    colors = ("red", "green", "blue", 'black', 'yellow', 'purple')
    # colors = plt.get_cmap('hsv', clus_l + 1)  # + 1 because of color[0] ~ color[n-1]
    # Create plot
    y1 = np.zeros(shape=(pop.mmm_inds.__len__()))
    y2 = np.zeros(shape=(pop.mmm_inds.__len__()))
    y3 = np.zeros(shape=(pop.mmm_inds.__len__()))
    y4 = np.zeros(shape=(pop.mmm_inds.__len__()))
    y5 = np.zeros(shape=(pop.mmm_inds.__len__()))
    y6 = np.zeros(shape=(pop.mmm_inds.__len__()))
    for i in range(pop.mmm_inds.__len__()):
        cur_inds = pop.mmm_inds[i]
        y1[i] = cur_inds[0].value
        y2[i] = cur_inds[1].value
        y3[i] = cur_inds[2].value
        y4[i] = cur_inds[3].value
        y5[i] = cur_inds[4].value
        y6[i] = cur_inds[5].value
    ax.plot(y1, c=colors[0], label='лучший архивный')
    ax.plot(y2, c=colors[1], label='средний архивный')
    ax.plot(y3, c=colors[2], label='худший архивный')
    ax.plot(y4, c=colors[3], label='лучший из популяции')
    ax.plot(y5, c=colors[4], label='средний из популяции')
    ax.plot(y6, c=colors[5], label='худший из популяции')
    plt.plot()
    title = 'Разнообразие популяции'
    plt.title(title)
    ax.set_xlabel('Поколение')
    ax.set_ylabel('Значение функции')
    plt.legend(loc=1)
    plt.grid(linestyle='--')
    plt.show()

    www = visualization.vis_main(pop.gen_history)
    global windows
    windows.append(www)



def n_runs(param_dict):
    # vars
    var_count = param_dict['variables count']
    param_dict['history'] = False
    run_count = param_dict['runs count']
    calc_count = param_dict['calculations count']
    f_name = param_dict['function name']

    # output
    # Initialize a workbook
    book = xlwt.Workbook(encoding="utf-8")
    # Add a sheet to the workbook
    sheet = book.add_sheet("N Runs")
    sheet.write(0, 0, 'Values')
    sheet.write(0, 1, 'Variables')
    # Run!

    pop = None

    if f_name == 'cassini2':
        print('matlab init')
        cassini2_problem.process_init()
        print('matlab init end!')

    for i in range(run_count):
        pop = ga.Population(param_dict)
        pop.evolve()
        best = pop.archive[0]
        print('Run №' + str(i + 1) + ' complete!')
        sheet.write(i + 1, 0, best.value.__str__())
        for j in range(var_count):
            sheet.write(i + 1, j + 1, best.variables[j].__str__())

    if f_name == 'cassini2':
        cassini2_problem.process_term()

    print('---_Done_---')
    # info
    now = datetime.datetime.now()
    ns = now.strftime("%Y-%m-%d %H:%M:%S")
    s_info = book.add_sheet("Info")

    s_info = book.add_sheet("Info")
    s_info.write(0, 0, "Date-time")
    s_info.write(0, 1, ns)

    i = 1
    for x in param_dict.keys():
        s_info.write(i, 0, str(x))
        i += 1
    i = 1
    for x in param_dict.values():
        s_info.write(i, 1, str(x))
        i += 1

    # alg_inf = "Shard Algorithm with selection and crossover, version 1.2, developed by Vasiliev Dmitriy, September 2019"
    # s_info.write(0, 3, alg_inf)

    book.save("output_n.xls")


def test_runs(param_dict):
    # TEST TEST TEST
    print('Test is on')

    '''
        param_dict['function name'] = f_name
        param_dict['is maximization'] = False
        param_dict['variables count'] = var_count
        param_dict['function borders'] = bord_all
        param_dict['calculations count'] = calc_count
        param_dict['runs count'] = runs_count
        param_dict['maximum individuals count'] = max_pop_count
        param_dict['initial individuals count'] = init_pop_count
        param_dict['archive individuals count'] = max_pop_count
        param_dict['mutation probability'] = mut_p
        param_dict['maximal life probability'] = max_lp
        param_dict['decay step'] = decay_step
        param_dict['archive resurrect probability'] = archive_p
        param_dict['archive use in crossover probability'] = archive_p
        param_dict['history'] = False
    '''
    run_count = 100
    init_count = 100
    max_count = 100
    mut_ch = 0.05

    param_dict['runs count'] = run_count
    param_dict['maximum individuals count'] = max_count
    param_dict['initial individuals count'] = init_count
    param_dict['mutation probability'] = mut_ch

    test_list = ['85 10']

    var_count = 0
    calc_count = 0
    for ttt in test_list:
        print(ttt)
        tt = ttt.split(' ')
        t1 = float(tt[0]) * 0.01
        t2 = float(tt[1]) * 0.01
        param_dict['maximal life probability'] = t1
        param_dict['decay step'] = t2
        for func_number in range(1, 9):
            func_name = ''
            if func_number == 1:
                func_name = 'sphere'
            elif func_number == 2:
                func_name = 'cigar'
            elif func_number == 3:
                func_name = 'rosenbrock'
            elif func_number == 4:
                func_name = 'ackley'
            elif func_number == 5:
                func_name = 'griewank'
            elif func_number == 6:
                func_name = 'rastrigin'
            elif func_number == 7:
                func_name = 'schaffer'
            elif func_number == 8:
                func_name = 'schwefel'
            for test_n in range(0, 3):
                if test_n == 0:
                    var_count = 2
                    calc_count = 5000
                elif test_n == 1:
                    var_count = 5
                    calc_count = 20000
                else:
                    # var_count = 20
                    var_count = 10
                    calc_count = 50000
                    # calc_count = 50000
                mess = func_name + '_v' + str(var_count)
                file_name = 'tests/' + ttt + '/EE_' + mess + '.xls'
                print(mess)
                bord_all = define_borders(func_name, var_count)
                # output
                # Initialize a workbook
                book = xlwt.Workbook(encoding="utf-8")
                # Add a sheet to the workbook
                sheet = book.add_sheet("N Runs")
                sheet.write(0, 0, 'Values')
                sheet.write(0, 1, 'Variables')
                # Run!

                param_dict['calculations count'] = calc_count
                param_dict['variables count'] = var_count
                param_dict['function name'] = func_name
                param_dict['function borders'] = bord_all

                pop = None

                if func_name == 'cassini2':
                    print('matlab init')
                    cassini2_problem.process_init()
                    print('matlab init end!')

                for i in range(run_count):
                    pop = ga.Population(param_dict)
                    pop.evolve()
                    best = pop.archive[0]
                    print('Run №' + str(i + 1) + ' complete!')
                    sheet.write(i + 1, 0, best.value.__str__())
                    for j in range(var_count):
                        sheet.write(i + 1, j + 1, best.variables[j].__str__())

                if func_name == 'cassini2':
                    cassini2_problem.process_term()

                print('---_Done_---')
                # info
                now = datetime.datetime.now()
                ns = now.strftime("%Y-%m-%d %H:%M:%S")
                s_info = book.add_sheet("Info")
                s_info.write(0, 0, "Date-time")
                s_info.write(0, 1, ns)

                i = 1
                for x in param_dict.keys():
                    s_info.write(i, 0, str(x))
                    i += 1
                i = 1
                for x in param_dict.values():
                    s_info.write(i, 1, str(x))
                    i += 1

                book.save(file_name)


start_window()
