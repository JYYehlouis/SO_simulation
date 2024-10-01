from typing import Tuple, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

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

def mask_margins(
    seq: npArrf64, sources: npArrf64, mask: npArrf64
) -> npArrf64:
    pass

def midpoint_algorithm(
    R: npf64, seq: npArrf64, sources: npArrf64, magnitude: npf64
) -> npArrf64:
    pass

def pupil_margins(
    seq: npArrf64, sources: npArrf64, pupil: str = 'circle'
) -> npArrf64:
    match pupil:
        case 'circle', '_':
            pass

if __name__ == "__main__":
    source_seq(1, 0.005, 10)