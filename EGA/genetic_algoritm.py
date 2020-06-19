import numpy as np
import random
import math
from decimal import Decimal
import func_deap
import cassini2_problem as cassini2_problem
import copy

func_name = "r"
glob_calc_count = 0


def g_ratio_parts(parts):
    perc = 0.62
    if parts == 1:
        return np.ndarray([1.0])
    a = np.zeros(shape=(parts))
    mod = 1.0
    for x in range(1, parts):
        b = mod * perc
        a[x - 1] = b
        mod -= b
    a[parts - 1] = mod
    return a


def calc_function(arr):
    global func_name
    val = None
    if func_name == 'sphere':
        val = func_deap.sphere(arr)
    elif func_name == 'cigar':
        val = func_deap.cigar(arr)
    elif func_name == 'rosenbrock':
        val = func_deap.rosenbrock(arr)
    elif func_name == 'ackley':
        val = func_deap.ackley(arr)
    elif func_name == 'griewank':
        val = func_deap.griewank(arr)
    elif func_name == 'rastrigin':
        val = func_deap.rastrigin(arr)
    elif func_name == 'schaffer':
        val = func_deap.schaffer(arr)
    elif func_name == 'schwefel':
        val = func_deap.schwefel(arr)
    elif func_name == 'cassini2':
        val = cassini2_problem.calc(arr.tolist())
        # print(arr)
        return val
    else:
        raise Exception('func_name variable is incorrect. Check list of available functions in description!')
    return val[0]


class Individual:
    def __init__(self, variables):
        self.variables = variables  # x,y,z,..
        self.value = calc_function(variables)
        global glob_calc_count
        glob_calc_count += 1

        # Don't forget to add this to .reboot()
        self.is_evaluate = False
        self.life_chance = None
        self.selection_rank = None
        self.lifetime = 0

        self.distance = None

    def tick(self, temp_modifier):
        # True if Alive
        r = random.uniform(0, 1)
        base_chance = copy.copy(self.life_chance)
        final_chance = base_chance + temp_modifier
        final_chance = min(1.0, final_chance)
        final_chance = max(0.01, final_chance)
        if r <= final_chance:
            return True
        else:
            return False

    def reboot(self):
        self.is_evaluate = False
        self.life_chance = None
        self.selection_rank = None
        self.lifetime = 0
        return self

    def __repr__(self):
        return "<Ind: value=%r ; vars=%r; life=%r; lifetime=%r>" % (self.value, self.variables, self.life_chance,
                                                                    self.lifetime)


def value_key(x):
    return x.value


def distance_key(x):
    return x.distance


# Classic DE scheme
# Loop until:
# 1. Selection
# 2. Crossover
# 3. Mutation
class Population:
    def __init__(self, param_dict):
        # golden ratio ~ 0.61803
        global func_name
        func_name = param_dict['function name']
        self.need_history = param_dict['history']
        self.individuals = list()  # refactor gen_step, info variables in lists (generations)
        self.var_count = param_dict['variables count']
        self.var_borders = param_dict['function borders']  # [ variable, 0 ] - min; [ variable, 1 ] - max
        self.max_population = param_dict['maximum individuals count']
        self.max_calculation_count = param_dict['calculations count']
        self.mutation_p = param_dict['mutation probability']
        self.mutation_p /= param_dict['variables count']
        self.calc_count = 0
        self.isMaximisation = param_dict['is maximization']
        init_count = param_dict['initial individuals count']
        # Init
        for x in range(init_count):
            ar = np.zeros(shape=(self.var_count))
            for v in range(self.var_count):
                r = random.uniform(self.var_borders[v][0], self.var_borders[v][1])
                ar[v] = r
            self.individuals.append(Individual(ar))
        self.archive = list()
        self.archive_max_count = param_dict['archive individuals count']
        self.gen_history = list()  # debug
        self.mmm_inds = list()  # min med max individuals
        if self.need_history:
            self.gen_history.append(self.individuals.copy())

        self.min_population = 6
        self.cur_population_count = init_count

        self.GR_parts = 3
        self.life_chance_list = list()
        self.life_decay_list = list()
        for i in range(self.GR_parts):
            p = param_dict['maximal life probability'] - abs(param_dict['decay step']) * i
            p = round(p, 3)
            self.life_chance_list.append(p)
            self.life_decay_list.append(p)
        # self.life_decay_list = [0.8, 0.7, 0.6]
        # self.life_decay_list = [0.99, 0.8, 0.7]
        # self.life_chance_list = [0.9, 0.8, 0.7]
        # self.life_chance_list = [0.99, 0.9, 0.8]
        self.rank_selection_chances = g_ratio_parts(self.GR_parts)

        self.eta_x = 20  # for SBCrossover
        self.eta_m = 20  # for Polynomial Mutation
        self.resurrect_chance = param_dict['archive resurrect probability']
        # self.resurrect_chance = 0.00
        self.archive_using_in_x_chance = param_dict['archive use in crossover probability']
        # self.archive_using_in_x_chance = 0.00

        # Variety
        self.use_variety = param_dict['use variety']
        # print('Use variety = ' + str(self.use_variety))
        self.distance_overflow_flag = False
        self.distance_memory_count = 5
        self.distance_array = np.zeros(shape=(self.distance_memory_count))
        self.sorting_balance_var = param_dict['sorting balance']  # 0. - only distance, 1. - only value

        # hybrid mut
        self.hybrid_mut_balance = param_dict['hybrid mutation balance']  # 0. - only D.E., 1. - only polynomial
        # DE
        self.is_DA = param_dict['is DA']
        # self.F = 0.5
        # self.p_of_best = 0.2
        self.archive_mut_chance = 0.2
        # self.archive_mut_chance = 0.0

        # Simple
        self.is_simple = False
        if self.is_simple:
            print('SIMPLE')
            self.archive_using_in_x_chance = 0.00
            self.resurrect_chance = 0.00
            self.archive_max_count = 1

            # self.life_decay_list = [1.0, 1.0, 1.0]
            # self.life_chance_list = [1.0, 1.0, 1.0]

        # additional variables
        self.AV_archive_use_count = 0
        self.AV_mutations_count = 0
        # self.AV_successfully_mutated_count = 0
        self.AV_mean_lifetime = 0  # For all population
        self.AV_mean_lt_calc_count = 0  # For all population

        self.is_log_on = param_dict['is log']

    def sort_by_distance(self, subpop, dist):
        lng = subpop.__len__()
        for x in range(lng):
            subpop[x].distance = dist[x]
        subpop = sorted(subpop, key=distance_key)
        subpop.reverse()
        return subpop

    def sort_by_value(self, subpop):
        subpop = sorted(subpop, key=value_key)
        if self.isMaximisation:
            subpop.reverse()
        return subpop

    def evolve(self):
        # Mutation - None
        # Crossover - Child Dotes?
        # Selection - Extinction
        step = 0
        while not self.check_stop():
            if self.is_log_on:
                print('cur step: ' + str(step))
                print('calculation count: ' + str(self.calc_count))
            self.perform_generation_step()
            if self.is_log_on:
                # print('pop count: ' + str(self.cur_population_count))
                print('value: ' + str(self.archive[0].value))
                print('------------------------')
            step += 1
        # post-calculations
        global glob_calc_count
        glob_calc_count = 0
        for x in self.individuals:
            self.rip_ind(x)
        self.AV_mean_lifetime /= self.AV_mean_lt_calc_count
        self.AV_mean_lifetime = round(self.AV_mean_lifetime, 3)

    def check_stop(self):
        if self.calc_count < self.max_calculation_count:
            return False
        else:
            return True

    def rip_ind(self, individual):
        # For statistic calculations
        self.AV_mean_lifetime += individual.lifetime
        self.AV_mean_lt_calc_count += 1

    def var_border_check(self, x, upper, lower):
        if x < lower:
            randperc = random.uniform(0.005, 0.02)
            step = (upper - lower) * randperc
            x = lower + step
        elif x > upper:
            randperc = random.uniform(0.005, 0.02)
            step = (upper - lower) * randperc
            x = upper - step
        '''
        Doen't work properly, slow as turtle
        while x < lower or x > upper:
            if x < lower:
                x += lower
                x /= 2
                x += 0.0000001  # just in case
            elif x > upper:
                x += upper
                x /= 2
                x -= 0.0000001  # just in case
        '''
        return x

    def mut_Gaussian(self, individual):
        print('GAUSS GAUSS GAUSS!')
        # size mu sigma
        mutp = self.mutation_p
        size = self.var_count
        mu = list()
        sigma = list()
        for i in range(size):
            mu.append(size)
            sigma.append(size)
        is_mut = False
        var_ar = np.copy(individual.variables)
        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < mutp:
                is_mut = True
                var_ar[i] += random.gauss(m, s)
                self.AV_mutations_count += 1
        if is_mut:
            return Individual(var_ar)
        else:
            return individual

    def mut_dif_ev(self, individual, pop, p_best):
        # mut chance???
        # ind check too
        donor = np.copy(individual.variables)
        # donor[i] = X[i] + F * (best_p[i] - X[i]) + F*(X_r1[i] - X_r2[i])
        ind_l = pop.__len__()
        # p = math.ceil(self.p_of_best * ind_l)
        # best_p = pop[random.randint(0, p)]
        best_p = pop[random.randint(0, p_best.__len__() - 1)]
        while_stopper = 0
        while self.compare_individuals(individual, best_p) == 0:
            # best_p = pop[random.randint(0, p)]
            best_p = pop[random.randint(0, p_best.__len__() - 1)]
            while_stopper += 1
            if while_stopper == 15:
                break
        is_a_used = False
        r1 = random.randint(0, ind_l - 1)
        x_r1 = pop[r1]
        while_stopper = 0
        while self.compare_individuals(individual, x_r1) == 0 or self.compare_individuals(best_p, x_r1) == 0:
            # 0 means equality
            r1 = random.randint(0, ind_l - 1)
            x_r1 = pop[r1]
            while_stopper += 1
            if while_stopper == 15:
                break

        if random.random() < self.archive_mut_chance and self.archive.__len__() > 0:
            r2 = random.randint(0, self.archive.__len__() - 1)
            x_r2 = self.archive[r2]
            is_a_used = True
        else:
            r2 = random.randint(0, ind_l - 1)
            x_r2 = pop[r2]

        while_stopper = 0
        while self.compare_individuals(individual, x_r1) == 0 or self.compare_individuals(best_p, x_r2) == 0 \
                or self.compare_individuals(x_r1, x_r2) == 0:
            is_a_used = False
            if random.random() < self.archive_mut_chance and self.archive.__len__() > 0:
                r2 = random.randint(0, self.archive.__len__() - 1)
                x_r2 = self.archive[r2]
                is_a_used = True
            else:
                r2 = random.randint(0, ind_l - 1)
                x_r2 = pop[r2]
            while_stopper += 1
            if while_stopper == 15:
                break
        best_p = np.copy(best_p.variables)
        x_r1 = np.copy(x_r1.variables)
        x_r2 = np.copy(x_r2.variables)
        if is_a_used:
            self.AV_archive_use_count += 1
        F = random.normalvariate(0.5, 0.2)
        F = max(F, 0)
        F = min(F, 1)
        # k for late [0,1] first or second sum
        for i in range(self.var_count):
            donor[i] += F * (best_p[i] - donor[i]) + F * (x_r1[i] - x_r2[i])
            donor[i] = self.var_border_check(donor[i], self.var_borders[i, 1], self.var_borders[i, 0])

        return Individual(donor)

    def mut_polynomial_mutation(self, individual, chance_mutation):
        # Author - K. Deb, Polynomial Mutation;
        # Origin of Python code: DEAP,  https://github.com/DEAP

        cent = 20
        std = 5
        eta = random.normalvariate(cent, std)
        eta = min(eta, cent + std)
        eta = max(eta, cent - std)

        new_ind_val = np.copy(individual.variables)
        is_mutated = False
        for i in range(self.var_count):
            if random.random() <= chance_mutation:
                is_mutated = True
                x = new_ind_val[i]
                x_lower = self.var_borders[i, 0]
                x_upper = self.var_borders[i, 1]

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
                x = self.var_border_check(x, x_upper, x_lower)
                new_ind_val[i] = x

        if is_mutated:
            # Compare mutant and original???
            new_ind = Individual(new_ind_val)
            new_ind.life_chance = individual.life_chance
            new_ind.selection_rank = individual.selection_rank
            new_ind.is_evaluate = individual.is_evaluate

            self.AV_mutations_count += 1

            return new_ind

        return individual

    def crossover_SBX(self, ind1, ind2):
        # Author - K. Deb, Simulated Binary Crossover;
        # Origin of Python code: DEAP,  https://github.com/DEAP

        cent = 20
        std = 5
        eta = random.normalvariate(cent, std)
        eta = min(eta, cent + std)
        eta = max(eta, cent - std)

        n_ind1 = np.copy(ind1.variables)
        n_ind2 = np.copy(ind2.variables)
        if random.random() < self.archive_using_in_x_chance:
            if self.archive.__len__() > 0:
                r_i = random.randint(0, self.archive.__len__() - 1)
                ind = self.archive[r_i].reboot()
                if random.random() <= 0.5:
                    n_ind1 = np.copy(ind.variables)
                else:
                    n_ind2 = np.copy(ind.variables)
                self.AV_archive_use_count += 1

        is_changed = False
        r_list = list()
        for i in range(self.var_count):
            if random.random() <= 0.5:
                is_changed = True
                r = True
            else:
                r = False
            r_list.append(r)
        if not is_changed:
            r_list[random.randint(0, self.var_count - 1)] = True

        for i in range(self.var_count):
            if r_list[i]:
                # may not crossover some variables
                # beta rnd: may crossover closer by himself, or parent???

                # This epsilon should probably be changed for 0 since
                # floating point arithmetic in Python is safer
                ind1_x = ind1.variables[i]
                ind2_x = ind2.variables[i]
                if abs(ind1_x - ind2_x) > 1e-14:
                    x1 = min(ind1_x, ind2_x)
                    x2 = max(ind1_x, ind2_x)
                    xl = self.var_borders[i, 0]
                    xu = self.var_borders[i, 1]
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
                    if beta_q == math.inf or beta_q == -math.inf:
                        print('Inf error!')
                        continue
                    c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                    # a border check
                    c1 = self.var_border_check(copy.copy(c1), copy.copy(xu), copy.copy(xl))
                    c2 = self.var_border_check(copy.copy(c2), copy.copy(xu), copy.copy(xl))

                    if random.random() <= 0.5:
                        n_ind1[i] = c2
                        n_ind2[i] = c1
                    else:
                        n_ind1[i] = c1
                        n_ind2[i] = c2

        # maybe return the best?
        # count it?
        r_ind = None
        if random.random() <= 0.5:
            r_ind = Individual(n_ind1)
        else:
            r_ind = Individual(n_ind2)
        # return r_ind
        return Individual(n_ind1), Individual(n_ind2)

    def compare_individuals(self, ind1, ind2):
        # 0 if ind1 = ind2
        # 10 if ind1.value = ind2.value, but vars not
        # 1 if ind1.value better than ind2.value
        # -1 if ind2.value better than ind1.value

        if self.isMaximisation:
            if ind1.value > ind2.value:
                return 1
            elif ind2.value > ind1.value:
                return -1
            else:
                is_same = True
                for x in range(self.var_count):
                    if ind1.variables[x] is not ind2.variables[x]:
                        is_same = False
                        break
                if is_same:
                    return 0
                else:
                    return 10
        else:
            if ind1.value < ind2.value:
                return 1
            elif ind2.value < ind1.value:
                return -1
            else:
                is_same = True
                for x in range(self.var_count):
                    if ind1.variables[x] != ind2.variables[x]:
                        is_same = False
                        break
                if is_same:
                    return 0
                else:
                    return 10

    def archive_individual(self, pop):
        lack = self.archive_max_count - self.archive.__len__()
        if lack > 0:
            # mesetaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaake!
            l = pop[0:lack]
            self.archive.extend(l)
        else:
            a_bot = self.archive[self.archive.__len__() - 1]
            i = 0
            # while pop[i] better or equal than a_bot, add him to buffer list
            app_arch = list()
            while not self.compare_individuals(pop[i], a_bot) == -1:
                app_arch.append(pop[i])
                i += 1
                if i >= pop.__len__():
                    break
            # add unique individuals to archive
            # compare to each other???????????????????????????
            unique = list()
            for x in app_arch:
                is_unique = True
                for y in self.archive:
                    if self.compare_individuals(x, y) != 1:
                        is_unique = False
                        break
                if is_unique:
                    unique.append(x)
            self.archive.extend(unique)
        self.archive = self.sort_by_value(self.archive)
        if self.archive.__len__() > self.archive_max_count:
            self.archive = self.archive[0:self.archive_max_count]

    def selection_DA(self, parent_list):
        child_list = list()
        p_best = []  # for DE mutation
        for x in parent_list:
            if x.selection_rank == 1:
                p_best.append(x)
        for x in parent_list:
            donor = self.mut_dif_ev(x, parent_list, p_best)
            c1, c2 = self.crossover_SBX(x, donor)
            child = None
            if random.random() > 0.5:
                child = c1
            else:
                child = c2
            child_list.append(child)
        return child_list

    def selection(self, parent_list):
        child_list = list()
        # Доделать селекцию, проверить, не хуже ли случайного выбора (сделать несколько) + переименовать блоки тогда
        # Ранговая?, пропорциональная?, и т.д., пока наверное по золотому сечению
        # что делоть со вторым индивидом????

        # 0 - random
        # 1 - golden ratio
        # 2 - Tournament
        flag = 2

        pop_len = parent_list.__len__()

        if flag == 2:
            winners = list()
            k = parent_list.__len__()  # !!!!!!!!!!!!!!
            # k = int(pop_len / 2)  # Child count
            if k % 2 == 1:
                k -= 1
            i = 0

            while i < k:
                asp1 = random.randint(0, pop_len - 1)
                asp2 = random.randint(0, pop_len - 1)
                comp = self.compare_individuals(parent_list[asp1], parent_list[asp2])
                if comp == 1:
                    winners.append(parent_list[asp1])
                    i += 1
                else:
                    winners.append(parent_list[asp2])
                    i += 1
            i = 0
            while i < k:
                c1, c2 = self.crossover_SBX(parent_list[i], parent_list[i + 1])
                child_list.append(c1)
                child_list.append(c2)
                i += 2
        else:
            for index in range(pop_len):
                ind1 = parent_list[index]
                ind2 = None
                if flag == 0:
                    # random
                    go = True
                    while go:
                        rand = random.randint(0, pop_len - 1)  # -1????
                        if index == rand:
                            continue
                        ind2 = parent_list[rand]
                        go = False
                elif flag == 1:
                    # golden ratio
                    go = True
                    while go:
                        rand = random.randint(0, pop_len - 1)  # -1????
                        if index == rand:
                            continue
                        ind2 = parent_list[rand]
                        try:
                            rank = ind2.selection_rank
                            p = self.rank_selection_chances[rank - 1]
                            if random.uniform(0, 1) <= p:
                                ind2 = parent_list[rand]
                                go = False
                        except:
                            # for resurection
                            pass
                else:
                    raise Exception("None of selection chosen. Check flag value")
                child = self.crossover_SBX(ind1, ind2)
                child_list.append(child)

        return child_list

    def r_calculation(self, ind1, ind2):
        r = 0.
        for i in range(self.var_count):
            xmin = self.var_borders[i][0]
            xmax = self.var_borders[i][1]
            val1 = (ind1.variables[i] - xmin) / (xmax - xmin)
            val2 = (ind2.variables[i] - xmin) / (xmax - xmin)
            r += pow(val1 - val2, 2)
        return math.sqrt(r)

    def add_distance(self, dist):
        if self.distance_overflow_flag == False:
            for x in range(self.distance_memory_count):
                if self.distance_array[x] == 0:
                    self.distance_array[x] = dist
                    if x == (self.distance_memory_count - 1):
                        self.distance_overflow_flag = True
                    return 0
        new_ar = np.zeros(shape=(self.distance_memory_count))
        for x in range(self.distance_memory_count - 1):
            new_ar[x] = self.distance_array[x + 1]
        new_ar[self.distance_memory_count - 1] = dist
        self.distance_array = new_ar

    def perform_generation_step(self):
        # 1. Set life time
        # 2. Selection
        # 3. Crossover
        # 4. Mutation
        # 5. Saving to Archive
        # 6. Extinction
        # 7. Extra-extinctions (if checked)
        # 8. Check, if population to low. True: extract individuals from Archive; False: pass.

        # ___________________________________
        # Step 1, Set life time:
        # Divide population in 3 ranks (parts), by golden ratio;
        # Set life chance for newborns, or decrease for others
        # corresponding to rank of individual
        # -----------------------------------

        pop_index_list = list()
        g_ratio = g_ratio_parts(self.GR_parts)
        rev_gr = g_ratio[::-1]
        left_for_last_part = self.cur_population_count
        # Individual.selected_rang - can use it???
        for x in range(self.GR_parts - 1):
            cur_pop_part = int(math.ceil(self.cur_population_count * rev_gr[x]))
            pop_index_list.append(cur_pop_part)
            left_for_last_part -= cur_pop_part
        pop_index_list.append(left_for_last_part)

        # Distance calculation
        perc_dist = None
        dist_list = None
        if self.use_variety:
            dist_list = np.zeros(shape=(self.individuals.__len__()))
            for i in range(self.individuals.__len__()):
                r_i = 0.
                for j in range(self.individuals.__len__()):
                    if i != j:
                        r_i += self.r_calculation(self.individuals[i], self.individuals[j])
                dist_list[i] = r_i

            mean_di = np.round(dist_list.mean(), 2)
            print('mean distance: ' + mean_di.__str__())
            if self.distance_overflow_flag:
                mean_mean_distance = self.distance_array.mean()
                dist_dif = mean_di - mean_mean_distance
                perc_dist = dist_dif / mean_mean_distance
                print('distance difference: ' + round(perc_dist, 3).__str__())
            self.add_distance(mean_di)
        is_dist_sort = False
        if random.random() > self.sorting_balance_var:
            if dist_list is not None:
                is_dist_sort = True
        populatiom = None
        if is_dist_sort:
            population = self.sort_by_distance(self.individuals, dist_list)
        else:
            population = self.sort_by_value(self.individuals)
        ind_pil = 0
        cur_part_index = pop_index_list[ind_pil]
        for index in range(self.cur_population_count):
            if index == cur_part_index:
                ind_pil += 1
                cur_part_index += pop_index_list[ind_pil]

            if not population[index].is_evaluate:
                population[index].life_chance = self.life_chance_list[ind_pil]
                population[index].is_evaluate = True
            else:
                population[index].life_chance *= self.life_decay_list[ind_pil]
            population[index].life_chance = round(population[index].life_chance, 5)
            population[index].selection_rank = ind_pil + 1  # Sic, ranges = [1,2,3]

        # ___________________________________
        # Step 2 and 3, Selection and Crossover:
        # not ready
        # SBX crossover, not ready!
        # -----------------------------------

        if (self.is_DA):
            # selection, crossover and mutation in 1 function, returns children
            children = self.selection_DA(population.copy())
            newpop = population.copy()
            newpop.extend(children)
        else:
            children = self.selection(population.copy())

            population.extend(children)

            # ___________________________________
            # Step 4, Mutation:
            # Mutate individuals
            # not ready!!!!!!!!!!!!!!!!
            # -----------------------------------

            newpop = list()
            p_best = []  # for DE mutation
            for x in population:
                if x.selection_rank == 1:
                    p_best.append(x)
            for x in population:
                if random.random() > self.hybrid_mut_balance:
                    # mut = self.mut_polynomial_mutation(x, self.mutation_p)
                    if random.random() < self.mutation_p * self.var_count:
                        mut = self.mut_dif_ev(x, population, p_best)
                        newpop.append(mut)
                    else:
                        newpop.append(x)
                else:
                    mut = self.mut_polynomial_mutation(x, self.mutation_p)
                    newpop.append(mut)
                # mut = self.mut_Gaussian(x)

        # ___________________________________
        # Step 4, Saving to Archive:
        # Add to list most valuable individuals
        # List of individuals must be sorted
        # -----------------------------------

        newpop = self.sort_by_value(newpop)
        self.archive_individual(newpop)

        # ___________________________________
        # Step 5, Extinction:
        # Check every individual life chance:
        # If individual successfully passed this check
        # he goes in next generation (maybe)
        # -----------------------------------

        # print('before tick pop count: ' + newpop.__len__().__str__())
        newpop2 = list()
        for x in newpop:
            # newborns can't die
            if x.is_evaluate:
                temp_modifier = 0.0
                if perc_dist is not None:
                    temp_modifier = copy.copy(perc_dist)
                    # temp_modifier /= 5
                    temp_modifier = min(temp_modifier, 0.15)
                    temp_modifier = max(temp_modifier, -0.15)
                    # temp_modifier = 0  # ppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp
                alive = x.tick(temp_modifier)
                if alive:
                    newpop2.append(x)
                else:
                    self.rip_ind(x)
            else:
                newpop2.append(x)

        # its time to stop! Code below need refactor

        # ___________________________________
        # Step 6, Extra-extinctions:
        # Check every individual life chance:
        # If individual successfully passed this check
        # he goes in next generation (maybe)
        # -----------------------------------

        # Nothing
        # How to add check???

        # Resurrection
        if random.random() <= self.resurrect_chance:
            r = random.randint(0, self.archive.__len__() - 1)

            chemzus = self.archive[r].reboot()
            newpop2.append(chemzus)
            self.AV_archive_use_count += 1

        # -----------------------------------
        # To the next generation!
        # -----------------------------------
        newpop2 = self.sort_by_value(newpop2)  # second sort only for chemzus
        # print('after tick pop count: ' + newpop2.__len__().__str__())

        pop = newpop2[0:self.max_population]

        # For statistic
        if newpop2.__len__() > self.max_population:
            dead = newpop2[self.max_population: newpop2.__len__()]
            for x in dead:
                pass
                # self.rip_ind(x)

        # Extraction from Archive
        if pop.__len__() < self.min_population:
            lack = self.min_population - pop.__len__()
            # print('lack : ' + str(lack))
            max_archive = self.archive.__len__()
            rand_l = random.sample(range(max_archive), lack)
            self.AV_archive_use_count += lack
            for x in rand_l:
                ind = self.archive[x].reboot()
                pop.append(ind)

        # For statistic
        for x in pop:
            x.lifetime += 1

        # End of generation
        self.individuals = pop
        self.cur_population_count = self.individuals.__len__()
        # Save each generation for plots
        if self.need_history:
            self.gen_history.append(self.individuals)
            if glob_calc_count > (self.max_calculation_count * 0.0):
                tmpl = list()
                cap_a = self.archive.__len__()
                cap_i = self.individuals.__len__()
                tmpl.append(self.archive[0])
                tmpl.append(self.archive[round(cap_a / 2)])
                tmpl.append(self.archive[cap_a - 1])
                tmpl.append(self.individuals[0])
                tmpl.append(self.individuals[round(cap_i / 2)])
                tmpl.append(self.individuals[cap_i - 1])
                self.mmm_inds.append(tmpl)

        self.calc_count = glob_calc_count

    def perform_simple_generation_step(self):
        children = self.selection(self.individuals.copy())
        newgen = list()
        for x in children:
            mut = self.mut_polynomial_mutation(x, self.mutation_p)
            newgen.append(mut)
        newgen = self.sort_by_value(newgen)
        if self.archive.__len__() == 0:
            self.archive.append(newgen[0])
        else:
            comp = self.compare_individuals(newgen[0], self.archive[0])
