import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Tuple

# constants
# c = 3e8
# epsilon = 8.85e-12
# mu = 4 * np.pi * 1e-7
eta = 377

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

def get_plane_wave(x: npt.ArrayLike, magnitude: np.float128 = 10) -> npt.ArrayLike:
    """
        Parameters:
            x (np.ndarray): a 1D array of x values
            magnitude (float): the magnitude of the plane

        Returns:
            plane (np.ndarray): a 1D array of y values
    """
    return magnitude * np.ones_like(x)

def generate_freq_from_xy(
    E_out: npt.ArrayLike, 
    size: int, 
    Fs: int
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
        Parameters:
            E_out (np.ndarray): a 1D array of y values
            size (int): the size of the input space
            Fs (int): the number of points in the sample

        Returns:
            x (np.ndarray): a 1D array of x values (frequent space)
            y (np.ndarray): a 1D array of y values (value)
    """
    N = (size - 1) // 2
    x = Fs / N * np.arange(-N, N + 1)
    y = np.concatenate((E_out[N + 1:], E_out[:N + 1]))
    return x, y


def generate_xy_from_freq(
    E_out: npt.ArrayLike, 
    size: int, 
    Fs: int
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
        Parameters:
            E_out (np.ndarray): a 1D array of y values
            size (int): the size of the input space
            Fs (int): the number of points in the sample

        Returns:
            x (np.ndarray): a 1D array of x values (x space)
            y (np.ndarray): a 1D array of y values (value)
    """
    N = (size - 1) // 2
    y = np.concatenate((E_out[N + 1:], E_out[:N + 1]))
    x = get_x(N, Fs)
    return x, y


def plotTransformation(space: int = 1000, width: int = 100, Fs: int = 100):
    """
        Plots the transformation of the plane wave

        Parameters:
            space (int): the range of x values
            width (int): the width of the mask in nanometers
            Fs (int): the number of points in the sample

        Returns: None
    """
    # x, mask, plot mask depening on x 
    x, mask = plot1dMask(space, width, Fs)

    # E_in before the mask, E_out after the mask, E_out = F{E_in * mask}
    # F{.} means Fourier Transform
    E_in, freq, E_out = plot1dFT(x, mask)

    # after aperture
    cut = 0.0025
    E_cut = plot1dCut(freq, E_out, cut)

    # inverse fft
    E_ifft = plot1dIFT(E_cut, x)

    return x, mask, E_in, freq, E_out, E_cut, E_ifft


def plot1dMask(
    space: int = 1000, 
    width: int = 100, 
    Fs: int = 100
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
        Plots the transformation of the plane wave

        Parameters:
            space (int): the range of x values
            width (int): the width of the mask in nanometers
            Fs (int): the number of points in the sample

        Returns: None
    """
    # x, mask, E_in 
    x = get_x(space, Fs)
    mask = get_mask(x, width)
    # set up the plot
    plt.figure(figsize=(8, 6))
    plt.title('Mask')
    plt.xlabel('x (nm)')
    plt.ylabel('Magnitude')
    plt.xlim(-space, space)
    plt.ylim(0, 1.1)
    plt.plot(x, mask, '-')
    plt.savefig('./img/plot1dMask.png')
    plt.show()
    
    return x, mask


def plot1dFT(
    x: npt.ArrayLike, 
    mask: npt.ArrayLike
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    E_in = get_plane_wave(x)
    E_out = np.fft.fft(np.multiply(mask, E_in))
    freq, E_out = generate_freq_from_xy(E_out, x.shape[0], Fs)
    cond = np.abs(freq) <= 0.1
    f = freq[cond]
    E = np.abs(E_out[cond])
    # I = np.power(E, 2)
    # set up the plot
    y_max = np.max(E)
    x_max = 0.1
    # plot Transformation
    plt.figure(figsize=(8, 6))
    plt.title('FT of a Plane Wave after a Mask')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Intensity')
    plt.ylim(0, y_max)
    plt.xlim(-x_max, x_max)
    plt.plot(f, E, '-')
    plt.savefig('./img/plot1dFT.png')
    plt.show()
    return E_in, freq, E_out


def plot1dCut(
    freq: npt.ArrayLike, 
    E_out: npt.ArrayLike, 
    cut: np.float128
) -> npt.ArrayLike:
    cond_cut = np.abs(freq) <= cut
    E_cut = np.where(cond_cut, E_out, 0)
    E = np.abs(E_cut)
    # I = np.power(E, 2)
    # set up the plot
    y_max = np.max(E)
    x_max = 0.1
    # plot Transformation
    plt.figure(figsize=(8, 6))
    plt.title('After Aperture')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Intensity')
    plt.ylim(0, y_max)
    plt.xlim(-x_max, x_max)
    plt.axline((-cut, 0), (-cut, y_max), color='red', linestyle='-')
    plt.axline((cut, 0), (cut, y_max), color='red', linestyle='-')
    plt.plot(freq, E, '-')
    plt.savefig('./img/plot1dCut.png')
    plt.show()
    return E_cut


def plot1dIFT(
    E_cut: npt.ArrayLike,
    x: npt.ArrayLike,
    I_min: np.float128 = 0.4
) -> npt.ArrayLike:
    E_ifft = np.fft.ifft(E_cut)
    E = np.abs(E_ifft)
    I = np.power(E, 2) / (2 * eta)
    # set up the plot
    y_max = np.max(I)
    x_max = np.max(x)
    # plot Transformation
    plt.figure(figsize=(8, 6))
    plt.title('Aerial Image after Inverse FFT')
    plt.xlabel('x (nm)')
    plt.ylabel('Intensity')
    plt.ylim(0, y_max)
    plt.xlim(-x_max, x_max)
    plt.axline((-x_max, I_min), (x_max, I_min), color='red', linestyle='-')
    plt.plot(x, I, '-')
    plt.savefig('./img/plot1dIFT.png')
    plt.show()
    return E_ifft


if __name__ == '__main__':
    space, width, Fs = 1000, 100, 100
    plotTransformation(space, width, Fs)
