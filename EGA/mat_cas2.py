import numpy as np
import cassini2_problem as cassini2_problem
import genetic_algoritm as ga
import datetime
import xlrd
import xlwt


def define_borders():
    bord_all = np.zeros(shape=(22, 2))
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


def param_dict_init():
    f_name = 'cassini2'
    var_count = 22
    bord_all = define_borders()
    #calc_count = 250000
    calc_count = 250000
    runs_count = 30
    max_pop_count = 250  # 250
    init_pop_count = 250
    mut_p = 0.1
    max_lp = 0.85
    decay_step = 0.10
    archive_p = 0.2

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
    param_dict['archive individuals count'] = max_pop_count
    # param_dict['archive individuals count'] = max_pop_count
    param_dict['mutation probability'] = mut_p
    param_dict['maximal life probability'] = max_lp
    param_dict['decay step'] = decay_step
    param_dict['archive resurrect probability'] = archive_p
    param_dict['archive use in crossover probability'] = archive_p
    param_dict['history'] = False
    param_dict['use variety'] = False
    param_dict['sorting balance'] = 1.   # 0. - only distance, 1. - only value
    param_dict['hybrid mutation balance'] = 0.5  # 0. - only D.E., 1. - only polynomial
    param_dict['is DA'] = False
    param_dict['is log'] = False
    return param_dict


def run_single():
    print('script begin')
    param_dict = param_dict_init()
    print('matlab init begin')
    cassini2_problem.process_init()
    print('matlab init end')
    print('GA begin')
    pop = ga.Population(param_dict)
    pop.evolve()
    print('GA end')
    cassini2_problem.process_term()
    print('matlab terminate')
    best = pop.archive[0]
    print('Archive best:')
    print(best)
    print('calculations count: ' + str(param_dict['calculations count']))
    print('script end')


def run_mult():
    print('script begin')
    param_dict = param_dict_init()
    runs = param_dict['runs count']
    var_count = param_dict['variables count']
    calc_count = param_dict['calculations count']
    now = datetime.datetime.now()
    ns = now.strftime("%Y_%m_%d-%H_%M_%S")
    book = xlwt.Workbook(encoding="utf-8")
    # Add a sheet to the workbook
    sheet = book.add_sheet("N Runs")
    sheet.write(0, 0, 'Values')
    sheet.write(0, 1, 'Variables')
    for x in range(runs):
        print('run № ' + str(x + 1))
        print('matlab init begin')
        cassini2_problem.process_init()
        print('matlab init end')
        print('GA begin')
        pop = ga.Population(param_dict)
        pop.evolve()
        print('GA end')
        cassini2_problem.process_term()
        print('matlab terminate')
        best = pop.archive[0]
        print('Archive best:')
        print(best)
        print('calculations count: ' + str(param_dict['calculations count']))

        print('Run №' + str(x + 1) + ' complete!')
        sheet.write(x + 1, 0, best.value.__str__())
        for j in range(var_count):
            sheet.write(x + 1, j + 1, best.variables[j].__str__())

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
    print('script end')
    book.save(ns + "_mruns.xls")


run_mult()
