from pylab import *

def swap_halves_2d(a):
    n = len(a)
    assert n % 2 == 0

    m = n // 2

    b = a
    b = np.r_[
        b[m:, :],
        b[:m, :],
    ]
    b = np.c_[
        b[:, m:],
        b[:, :m],
    ]

    return b


def left_right_pad_2d(a, m):
    n = len(a)

    assert n % 2 == 0
    assert m % 2 == 0
    assert m >= n

    l = (m - n) // 2
    r = l + n

    b = np.zeros((m, m), dtype=a.dtype)
    b[l:r, l:r] = a

    return b


def left_right_unpad_2d(b, n):
    m = len(b)

    assert n % 2 == 0
    assert m % 2 == 0
    assert m >= n

    l = (m - n) // 2
    r = l + n

    return b[l:r, l:r]


def finite_fft_2d(n, a_f, step_f, ys_f_shifted, m):
    assert n % 2 == 0
    assert m % 2 == 0

    fft_arg = ys_f_shifted
    fft_arg = left_right_pad_2d(fft_arg, m)
    fft_arg = swap_halves_2d(fft_arg)

    fft_res = np.fft.fft2(fft_arg)
    ys_F = fft_res * step_f ** 2
    ys_F = swap_halves_2d(ys_F)
    ys_F = left_right_unpad_2d(ys_F, n)

    return ys_F


def finite_integral_2d(n, step_f, xs_f, ys_f, xs_F):
    shape = (n, n, n, n)

    # first dimension - x
    x_4d = np.broadcast_to(xs_f[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # second dimension - y
    y_4d = np.broadcast_to(xs_f[np.newaxis, :, np.newaxis, np.newaxis], shape)

    # third dimension - u
    u_4d = np.broadcast_to(xs_F[np.newaxis, np.newaxis, :, np.newaxis], shape)
    # forth dimension - v
    v_4d = np.broadcast_to(xs_F[np.newaxis, np.newaxis, np.newaxis, :], shape)

    # exp values
    A = np.exp((-2 * np.pi * 1j) * (x_4d * u_4d + y_4d * v_4d))

    # scale d1 and d2 by f(x, y)
    A = A * np.broadcast_to(ys_f[:, :, np.newaxis, np.newaxis], shape)

    int_weights = np.ones(n)
    int_weights[0] = 1 / 2
    int_weights[-1] = 1 / 2
    int_weights *= step_f

    # scale d1 by int_weights
    A = A * np.broadcast_to(int_weights[:, np.newaxis, np.newaxis, np.newaxis], shape)
    # scale d2 by int_weights
    A = A * np.broadcast_to(int_weights[np.newaxis, :, np.newaxis, np.newaxis], shape)

    ys_F = A
    ys_F = np.sum(ys_F, axis=0)
    ys_F = np.sum(ys_F, axis=0)

    return ys_F


def draw_2d(sp_n, sp_m, sp_c, xs, ys, s):
    extent = [xs[0], xs[-1], xs[0], xs[-1]]

    subplot(sp_n, sp_m, sp_c + 1)
    imshow(np.abs(ys), extent=extent)
    colorbar()
    title(f'|{s}|')

    subplot(sp_n, sp_m, sp_c + 2)
    imshow(np.angle(ys), extent=extent, vmin=-np.pi, vmax=np.pi)
    colorbar()
    title(f'angle {s}')

    subplot(sp_n, sp_m, sp_c + 3)
    imshow(np.real(ys), extent=extent)
    colorbar()
    title(f'Real {s}')

    subplot(sp_n, sp_m, sp_c + 4)
    imshow(np.imag(ys), extent=extent)
    colorbar()
    title(f'Imaginary {s}')