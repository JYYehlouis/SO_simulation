from typing import Tuple, List, Any, Callable
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import torch

npf64 = np.float64
npArrf64 = npt.NDArray[npf64]

def source_seq(
    s_max: npf64, f_max: npf64, samp1side: int, magnitude: int = 0
) -> Tuple[npArrf64, npArrf64, npArrf64]:
    """
        Parameters
        ----------
        s_max: npf64
            Maximum space of the spatial domain
        f_max: npf64
            Maximum frequency of the source
        samp1side: int
            Number of samples on one side of the source
        
        Returns
        ----------
        npArrf64
            Sequence of spatial space
        npArrf64
            Sequence of frequency space
        pArrf64
            Sequence of source in which the sources can be collected
    """
    samp = 2 * samp1side + 1
    s_seq = np.linspace(-s_max, s_max, samp)
    f_seq = np.linspace(-f_max, f_max, samp)
    sources = magnitude * np.ones(shape=(samp, samp))
    return s_seq, f_seq, sources

def generate_square_mask(
    s_space: npArrf64,
    W: int
) -> npArrf64:
    s_mask = np.where(np.abs(s_space) <= W, 1, 0)
    xx, yy = np.meshgrid(s_mask, s_mask)
    ret = xx & yy
    return ret

def generate_pupil(
    xx_freq: npArrf64, 
    yy_freq: npArrf64,
    R: npf64
) -> npArrf64:
    R_square = R ** 2
    ret_freq = xx_freq ** 2 + yy_freq ** 2
    ret_freq = np.where(np.abs(ret_freq) <= R_square, 1, 0)
    return ret_freq

def generate_Ein(
    K: npf64, 
    x: npArrf64,
    sinx: npf64,
    y: npArrf64,
    siny: npf64,
    magnitude: int = 1
) -> npArrf64:
    return magnitude * np.exp(1j * K * (x * sinx + y * siny))

def fourier_transform(
    arr: npArrf64
) -> npArrf64:
    return np.fft.fft2(arr)

def inv_fourier_transform(
    arr: npArrf64
) -> npArrf64:
    return np.fft.ifft2(arr)

def shift2D(arr: npArrf64) -> npArrf64:
    N = (arr.shape[0] - 1) // 2
    new_arr1 = np.zeros_like(arr)
    new_arr2 = np.zeros_like(arr)
    new_arr1[:N, :], new_arr1[N:, :] = arr[N + 1:, :], arr[:N + 1, :]
    new_arr2[:, :N], new_arr2[:, N:] = new_arr1[:, N + 1:], new_arr1[:, :N + 1]
    return new_arr2

def shift2Dinv(arr: npArrf64) -> npArrf64:
    N = (arr.shape[0] - 1) // 2
    new_arr1 = np.zeros_like(arr)
    new_arr2 = np.zeros_like(arr)
    new_arr1[:, :N + 1], new_arr1[:, N + 1:] = arr[:, N:], arr[:, :N]
    new_arr2[:N + 1, :], new_arr2[N + 1:, :] = new_arr1[N:, :], new_arr1[:N, :]
    return new_arr2

def T_sigmoid(THESHOLD: npf64, *args):
    a = 90
    if len(args) == 1: # mask itself (I)
        return 1 / (1 + np.exp(-a * (args[0].flatten() - THESHOLD)))
    else:
        return 1 / (1 + np.exp(-a * (np.dot(args[0], args[1]) - THESHOLD)))

def FR_sigmoid(TIt, TI):
    return np.linalg.norm(TIt - TI) ** 2

def SD_optimization(
    THESHOLD: npf64, I_dist: npArrf64, TIt: npArrf64, J: npArrf64, a: npf64,
    T_func: Callable[[npf64, Any], npArrf64], FR_func: Callable[[npArrf64, npArrf64], npf64], 
    gamma: Tuple[npf64, None], alpha: Tuple[npf64, None], tol: npf64 = 1e-4
) -> Tuple[npArrf64, List]:
    J_k = J
    gamma_k = gamma
    loss = []
    while 1:
        TI = T_func(THESHOLD, I_dist, J_k)
        g_k = -2 * a * np.dot(I_dist.T, (TIt - TI) * (1 - TI) * TI)
        g_k = g_k / np.linalg.norm(g_k)
        J_kp1 = J_k - gamma_k * g_k
        # set all negatives to 0
        J_kp1[J_kp1 < 0] = 0
        err = FR_func(TIt, TI)
        loss.append(err)
        # print(err)
        fr1, fr2 = err, FR_func(TIt, T_func(THESHOLD, I_dist, J_kp1))
        if fr2 > fr1:
            gamma_k = alpha * gamma_k
        elif fr1 - fr2 < tol:
            J_k = J_kp1
            break
        else:
            J_k = J_kp1
    return J_k, loss



