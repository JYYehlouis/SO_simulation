import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Tuple

def get_x(space: int, Fs: int) -> npt.ArrayLike:
    """
        Parameters:
            space (int): the range of x values
            sample (int): the number of points in the sample

        Returns:
            x (np.ndarray): a 1D array of x values
    """
    return np.linspace(-space, space, 2 * space * Fs + 1)

def get_mask(x: npt.ArrayLike, width) -> npt.ArrayLike:
    """
        Parameters:
            width (int): the width of the mask in nanometers
            n (int): the number of points in the mask

        Returns:
            mask (np.ndarray): a 1D array of 1's and 0's
    """
    return np.where(np.abs(x) <= width, 1, 0)

def get_plane_wave(x: npt.ArrayLike, magnitude: np.float128 = 1) -> npt.ArrayLike:
    """
        Parameters:
            x (np.ndarray): a 1D array of x values
            magnitude (float): the magnitude of the plane

        Returns:
            plane (np.ndarray): a 1D array of y values
    """
    return magnitude * np.ones_like(x)

def generate_xy(E_out: npt.ArrayLike, size: int, Fs: int) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
        Parameters:
            E_out (np.ndarray): a 1D array of y values
            size (int): the size of the input space

        Returns:
            x (np.ndarray): a 1D array of x values (frequent space)
            y (np.ndarray): a 1D array of y values (value)
    """
    N = (size - 1) // 2
    x = Fs / N * np.arange(-N, N + 1)
    y = np.concatenate((E_out[N + 1:], E_out[:N + 1]))
    return x, y

def plotTransformation():
    """
        Plots the transformation of the plane wave
    """
    space = 1000
    width = 100
    Fs = 100
    T = 1 / Fs
    # x, mask, E_in 
    x = get_x(space, Fs)
    mask = get_mask(x, width)
    E_in = get_plane_wave(x)
    # after the mask
    E_out = np.abs(np.multiply(E_in, np.fft.fft(mask)))
    freq, E_out = generate_xy(E_out, x.shape[0], Fs)
    cond = np.abs(freq) <= 0.1
    f = freq[np.abs(freq) <= 0.1]
    E = E_out[cond]
    # set up the plot
    y_max = np.max(E)
    x_max = 0.1
    # plot mask
    plt.figure(figsize=(6, 4))
    plt.title('Mask')
    plt.xlabel('x (nm)')
    plt.ylabel('Magnitude')
    plt.xlim(-space, space)
    plt.ylim(0, 1.1)
    plt.plot(x, mask, '-')
    plt.savefig('./img/plot1dMask.png')
    plt.show()
    # plot Transformation
    plt.figure(figsize=(8, 6))
    plt.title('Transformation of a Plane Wave after a Mask')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude')
    plt.ylim(0, y_max)
    plt.xlim(-x_max, x_max)
    plt.plot(f, E, '-')
    plt.savefig('./img/plot1dTransformation.png')
    plt.show()

if __name__ == '__main__':
    plotTransformation()