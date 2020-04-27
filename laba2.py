import numpy as np
from pylab import *
import itertools
from lab2 import *
from lab2d import *


def two_d():
    def func1():
        # gauss

        n = 1 << 6  # 2^6
        m = 1 << 8  # 2^8

        a_f = 5

        f_2d = lambda a: np.exp(-(a[:, :, 0] ** 2 + a[:, :, 1] ** 2))
        return f_2d, n, m, a_f

    def func2():

        n = 1 << 6  # 2^6
        m = 1 << 8  # 2^8

        a_f = 5
        f_2d = lambda a: np.sin(3 * np.pi * a[:, :, 0]) * np.sin(3 * np.pi * a[:, :, 1])

        return f_2d, n, m, a_f

    def calculate(f_2d, n, m, a_f):
        a_F = n ** 2 / (4 * a_f * m)

        step_f = 2 * a_f / (n - 1)
        step_F = 2 * a_F / (n - 1)

        xs_f = np.linspace(-a_f, a_f, n)
        xs_f_shifted = xs_f - step_f / 2
        xs_F = np.linspace(-a_F, a_F, n)

        Xs_f = np.reshape(list(itertools.product(xs_f, xs_f)), (n, n, 2))
        Xs_f_shifted = np.reshape(list(itertools.product(xs_f_shifted, xs_f_shifted)), (n, n, 2))
        Xs_F = np.reshape(list(itertools.product(xs_F, xs_F)), (n, n, 2))

        ys_f = ascomplex(f_2d(Xs_f))
        ys_f_shifted = ascomplex(f_2d(Xs_f_shifted))

        # fft
        ys_F_fft = ascomplex(finite_fft_2d(n, a_f, step_f, ys_f_shifted, m))

        # integral
        ys_F_integral = ascomplex(finite_integral_2d(n, step_f, xs_f, ys_f, xs_F))

        figure(figsize=(16, 8))
        draw_2d(4, 4, 0, xs_f, ys_f, 'f')
        draw_2d(4, 4, 4, xs_F, ys_F_fft, 'F_{fft}')
        draw_2d(4, 4, 8, xs_F, ys_F_integral, 'F_{integral}')
        show()

    #f_2d, n, m, a_f = func1()
    #calculate(f_2d, n, m, a_f)  # just exp
    f_2d, n, m, a_f = func2()
    calculate(f_2d, n, m, a_f)  # my function


def one_d():
    import numpy as np

    def func1():
        # gauss

        f = lambda x: np.exp(-x ** 2)

        n = 1 << 10  # 2 ^ 10
        m = 1 << 15  # 2 ^ 15

        a_f = 5
        return f, n, m, a_f

    def func2():

        f = lambda x: np.sin(3 * np.pi * x)

        n = 1 << 10  # 2 ^ 10
        m = 1 << 15  # 2 ^ 15

        a_f = 5
        return f, n, m, a_f

    def calculate(f, n, m, a_f):
        # prep
        a_F = n ** 2 / (4 * a_f * m)

        step_f = 2 * a_f / (n - 1)
        step_F = 2 * a_F / (n - 1)

        xs_f = np.linspace(-a_f, a_f, n)
        xs_f_shifted = xs_f - step_f / 2
        xs_F = np.linspace(-a_F, a_F, n)

        ys_f = ascomplex(f(xs_f))
        ys_f_shifted = ascomplex(f(xs_f_shifted))

        # fft
        ys_F_fft = ascomplex(finite_fft(n, a_f, step_f, ys_f_shifted, m))

        # integral
        ys_F_integral = ascomplex(finite_integral(n, step_f, xs_f, ys_f, xs_F))

        figure(figsize=(16, 8))
        draw_1d(4, 4, 0, xs_f, ys_f, 'f')
        draw_1d(4, 4, 4, xs_F, ys_F_fft, 'F_{fft}')
        draw_1d(4, 4, 8, xs_F, ys_F_integral, 'F_{integral}')
        show()

    #f, n, m, a_f = func1()
    #calculate(f, n, m, a_f)
    f, n, m, a_f = func2()
    calculate(f, n, m, a_f)

one_d()
two_d()
