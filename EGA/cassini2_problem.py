from __future__ import print_function
import cassini2_eta
import matlab

Cas_process = None


def process_init():
    global Cas_process
    Cas_process = cassini2_eta.initialize()


def process_term():
    global Cas_process
    Cas_process.terminate()


def calc(vector, problem=None):
    tIn = matlab.double(vector, size=(1, 22))
    problemIn = {"sequence": matlab.double([3.0, 2.0, 2.0, 3.0, 5.0, 6.0], size=(1, 6)),
                 "objective": {"type": "total DV rndv"},
                 "bounds": {
                     "lower": matlab.double([-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01,
                                             0.01, 0.01, 1.05, 1.05, 1.15, 1.7, -3.141592653589793, -3.141592653589793,
                                             -3.141592653589793, -3.141592653589793], size=(1, 22)),
                     "upper": matlab.double([0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9,
                                             0.9, 0.9, 6.0, 6.0, 6.5, 291.0, 3.141592653589793, 3.141592653589793,
                                             3.141592653589793, 3.141592653589793], size=(1, 22))},
                 "yplot": matlab.double([0.0], size=(1, 1))}
    output_value = Cas_process.cassini2(tIn, problemIn)
    return output_value
