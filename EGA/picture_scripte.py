import numpy as np
import random
import matplotlib.pyplot as plt


def mut_polynomial_mutation(individual, eta):
    # Author - K. Deb, Polynomial Mutation;
    # Origin of Python code: DEAP,  https://github.com/DEAP

    chance_mutation = 1

    new_ind_val = np.copy(individual)
    is_mutated = False
    for i in range(1):
        if random.random() <= chance_mutation:
            is_mutated = True
            x = new_ind_val[i]
            x_lower = -2
            x_upper = 2

            # somehow it changes
            delta_1 = (x - x_lower) / (x_upper - x_lower)
            delta_2 = (x_upper - x) / (x_upper - x_lower)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (x_upper - x_lower)
            new_ind_val[i] = x
        return new_ind_val

    return individual


def crossover_SBX(ind1, ind2, eta):
    # Author - K. Deb, Simulated Binary Crossover;
    # Origin of Python code: DEAP,  https://github.com/DEAP

    n_ind1 = np.copy(ind1)
    n_ind2 = np.copy(ind2)

    for i in range(1):
        # may not crossover some variables
        # beta rnd: may crossover closer by himself, or parent???

        # This epsilon should probably be changed for 0 since
        # floating point arithmetic in Python is safer
        ind1_x = ind1[i]
        ind2_x = ind2[i]
        if abs(ind1_x - ind2_x) > 1e-14:
            x1 = min(ind1_x, ind2_x)
            x2 = max(ind1_x, ind2_x)
            xl = -6
            xu = 6
            rand = random.random()

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            e_st = (1.0 / (eta + 1))  # !
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** e_st  # Разделить?
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** e_st

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            # - inf error
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            if random.random() <= 0.5:
                n_ind1[i] = c2
                n_ind2[i] = c1
            else:
                n_ind1[i] = c1
                n_ind2[i] = c2

    return n_ind1, n_ind2


def ret_ar(interv, ar, r_min, r_max):
    i_step = (r_max - r_min) / interv
    intervals_l_borders = [r_min + (i_step * x) for x in range(interv)]
    int_n = np.zeros(shape=(interv), dtype=np.int)
    for i in range(ar.shape[0]):
        for x in range(interv):
            if intervals_l_borders[x] <= ar[i] < (intervals_l_borders[x] + i_step):
                int_n[x] += 1
                break
    x_ar = np.zeros(shape=(interv))
    for i in range(interv):
        v = (intervals_l_borders[i] + i_step) / 2
        x_ar[i] = v
    return x_ar, int_n


def test2():
    v1 = np.array([-3.0])
    v2 = np.array([3.0])
    t_count = 100000
    muts_20 = np.zeros(shape=(t_count))
    muts_5 = np.zeros(shape=(t_count))
    muts_40 = np.zeros(shape=(t_count))
    for i in range(t_count):
        c1, _ = crossover_SBX(v1, v2, 20)
        c2, _ = crossover_SBX(v1, v2, 5)
        c3, _ = crossover_SBX(v1, v2, 40)
        muts_20[i] = c1[0]
        muts_40[i] = c3[0]
        muts_5[i] = c2[0]
    interv = 200
    r_min = -6
    r_max = 6
    x_ar1, n_ar1 = ret_ar(interv, muts_20, r_min, r_max)
    x_ar2, n_ar2 = ret_ar(interv, muts_5, r_min, r_max)
    x_ar3, n_ar3 = ret_ar(interv, muts_40, r_min, r_max)
    figures, axes = plt.subplots()
    axes.plot(x_ar3, n_ar3, color="red", label="eta = 40")
    axes.plot(x_ar1, n_ar1, color="blue", label="eta = 20")
    axes.plot(x_ar2, n_ar2, color="black", label="eta = 5")
    axes.set_title('SBX')
    axes.set_xlabel('x')
    axes.set_ylabel('n')
    axes.legend()
    plt.grid(True)
    plt.show()
    print('end')


def test():
    or_mut = np.zeros(shape=(1))
    t_count = 100000
    muts_20 = np.zeros(shape=(t_count))
    muts_5 = np.zeros(shape=(t_count))
    muts_40 = np.zeros(shape=(t_count))
    for i in range(t_count):
        m20 = mut_polynomial_mutation(or_mut, 20)
        m5 = mut_polynomial_mutation(or_mut, 5)
        m40 = mut_polynomial_mutation(or_mut, 40)
        muts_20[i] = m20[0]
        muts_40[i] = m40[0]
        muts_5[i] = m5[0]
    interv = 200
    r_min = -2
    r_max = 2
    x_ar1, n_ar1 = ret_ar(interv, muts_20, r_min, r_max)
    x_ar2, n_ar2 = ret_ar(interv, muts_5, r_min, r_max)
    x_ar3, n_ar3 = ret_ar(interv, muts_40, r_min, r_max)
    figures, axes = plt.subplots()
    axes.plot(x_ar3, n_ar3, color="red", label="eta = 40")
    axes.plot(x_ar1, n_ar1, color="blue", label="eta = 20")
    axes.plot(x_ar2, n_ar2, color="black", label="eta = 5")
    axes.set_title('Полиномиальная мутация')
    axes.set_xlabel('x')
    axes.set_ylabel('n')
    axes.legend()
    plt.grid(True)
    plt.show()
    print('end')


test2()
