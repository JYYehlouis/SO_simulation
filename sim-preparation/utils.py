from typing import Tuple, List
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

def midpoint_algo(samp1side: int, maxSize: int) -> List[Tuple[int, int]]:
    """
        A midpoint algorithm to find points on a circle
        
        Parameters
        ----------
        samp1side: int
            the number of sampling, the center and the radius of the circle

        Returns
        ----------
        List[Tuple[int, int]]
            A list contains points on the path of the circle with radius R
    """
    x_center, y_center, R = samp1side, samp1side, samp1side
    # valid point list
    ret: List[Tuple[int, int]] = []
    # initial point(s)
    currX, currY = R, 0
    # p-value to calculate if points are in the circle
    pVal, nextpVal = 0, 1.25 - R
    while currX > currY:
        # update the pVal
        pVal = nextpVal
        # use pVal to see the position of the midpoint 
        # and update nextpVal for the next midpoint
        
        # if the midpoint is in the circle
        # means the value somehow determine by (currentX, currentY)
        if pVal <= 0:
            # under this condition we know that 
            # the point (currX - 0.5, currY) is in the circle of radius R
            # update the pVal to nextpVal with (currX - 0.5, currY + 1)
            nextpVal = nextpVal + 2 * currY + 1
        else:
            # under this condition we know that
            # the point (currX, currY) is outside the circle of radius R
            # update the value to (currX - 1, currY + 1)
            # for convenience we use a line to implement
            break
        #    p5 p3
        # p7       p1
        # p8       p2
        #    p6 p4 
        if currY == 0:
            ret.append((currX + x_center, currY + y_center))    # p1
            ret.append((currY + x_center, currX + y_center))    # p3
            ret.append((currY + x_center, -currX + y_center))   # p4
            ret.append((-currX + x_center, currY + y_center))   # p7
        else:
            ret.append((currX + x_center, currY + y_center))    # p1
            ret.append((currX + x_center, -currY + y_center))   # p2
            ret.append((currY + x_center, currX + y_center))    # p3
            ret.append((currY + x_center, -currX + y_center))   # p4
            ret.append((-currY + x_center, currX + y_center))   # p5
            ret.append((-currY + x_center, -currX + y_center))  # p6
            ret.append((-currX + x_center, currY + y_center))   # p7
            ret.append((-currX + x_center, -currY + y_center))  # p8
        # the next valid points by increasing 1 on y
        currY += 1
    # ----------------------------------------
    # linear approach
    x, y = currX, currY
    while 1:
        if len(ret) == maxSize:
            break
        x -= 1
        y += 1
        #    p5 p3
        # p7       p1
        # p8       p2
        #    p6 p4
        if (x + x_center, y + y_center) in ret:
            break
        ret.append((x + x_center, y + y_center))    # p1
        ret.append((x + x_center, -y + y_center))   # p2
        ret.append((y + x_center, x + y_center))    # p3
        ret.append((y + x_center, -x + y_center))   # p4
        ret.append((-y + x_center, x + y_center))   # p5
        ret.append((-y + x_center, -x + y_center))  # p6
        ret.append((-x + x_center, y + y_center))   # p7
        ret.append((-x + x_center, -y + y_center))  # p8
    # ----------------------------------------
    return ret

def xypos_to_idx(positions: List[Tuple[int, int]]) -> List[int]:
    ret = []
    for pos in positions:
        ret.append(101 * pos[0] + pos[1])
    ret.sort()
    return ret

def square_mask_inside(width: int, samp1side: int, d_s: npf64) -> Tuple[List[Tuple[int]], int]:
    # the center of the circle
    x_center = y_center = samp1side
    # first check the boundary region of the mask
    x = 0
    while x * d_s < width:
        x += 1
    ret = []
    y = 0
    while y <= x:
        if y == 0:
            ret.append((x + x_center, y + y_center))    # p1
            ret.append((y + x_center, x + y_center))    # p3
            ret.append((y + x_center, -x + y_center))   # p4
            ret.append((-x + x_center, y + y_center))   # p7
        elif x == y:
            ret.append((x + x_center, y + y_center))
            ret.append((x + x_center, -y + y_center))
            ret.append((-x + x_center, y + y_center))
            ret.append((-x + x_center, -y + y_center))
        else:
            ret.append((x + x_center, y + y_center))    # p1
            ret.append((x + x_center, -y + y_center))   # p2
            ret.append((y + x_center, x + y_center))    # p3
            ret.append((y + x_center, -x + y_center))   # p4
            ret.append((-y + x_center, x + y_center))   # p5
            ret.append((-y + x_center, -x + y_center))  # p6
            ret.append((-x + x_center, y + y_center))   # p7
            ret.append((-x + x_center, -y + y_center))  # p8
        y += 1
    return ret, x

def inside_revise_for_opt(l: int, pos: List[int]) -> List[int]:
    rep1, rep2 = pos[0], pos[2 * l]
    rep3, rep4 = pos[-2 * l - 1], pos[-1]
    pos.insert(2 * l + 1, rep1)
    pos.insert(2 * l + 2, rep2)
    pos.insert(-2 * l - 1, rep4)
    pos.insert(-2 * l - 2, rep3)
    return pos
    

def inside_pixels_to_outside(width: int, samp1side: int, d_s: npf64) -> List[Tuple[int, int]]:
    # the center of the circle
    x_center = y_center = samp1side
    # first check the boundary region of the mask
    x = 0
    while x * d_s <= width:
        x += 1
    ret = []
    y = 0
    while y < x:
        if y == 0:
            ret.append((x + x_center, y + y_center))    # p1
            ret.append((y + x_center, x + y_center))    # p3
            ret.append((y + x_center, -x + y_center))   # p4
            ret.append((-x + x_center, y + y_center))   # p7
        else:
            ret.append((x + x_center, y + y_center))    # p1
            ret.append((x + x_center, -y + y_center))   # p2
            ret.append((y + x_center, x + y_center))    # p3
            ret.append((y + x_center, -x + y_center))   # p4
            ret.append((-y + x_center, x + y_center))   # p5
            ret.append((-y + x_center, -x + y_center))  # p6
            ret.append((-x + x_center, y + y_center))   # p7
            ret.append((-x + x_center, -y + y_center))  # p8
        y += 1
    return ret

def optimization(Q: npArrf64, b: npArrf64):
    J = np.zeros(shape=(Q.shape[1], 1))
    g = 2 * (Q * J - b)
    if np.linalg.norm(g) < 1e-4:
        return J
    d = -g
    while 1:
        gd = g.transpose() * d
        dtqd = d.transpose() * Q * d
        alpha = - gd / dtqd
        J = J + alpha * d
        # set all negatives to 0
        cond = np.where(J < 0, True, False)
        J[cond] = 0
        # end set
        g = 2 * (Q * J - b)
        err = np.linalg.norm(g)
        print(err)
        if err < 1e-4:
            return J
        beta = g.transpose() * Q * d / dtqd
        d = -g + beta * d

def optimization_torch(Q: npArrf64, b: npArrf64):
    if torch.cuda.is_available() == False:
        return None
    
    J = np.zeros(shape=(Q.shape[1], 1))
    torch_J = torch.from_numpy(J).to(torch.device('cuda'))
    torch_Q = torch.from_numpy(Q).to(torch.device('cuda'))
    torch_b = torch.from_numpy(b).to(torch.device('cuda'))
    torch_g = (2 * (torch.mm(torch_Q, torch_J) - torch_b)).to(torch.device('cuda'))
    torch_d = (-torch_g).to(torch.device('cuda'))
    while 1:
        torch_gd = torch.transpose(torch_g, 0, 1).to(torch.device('cuda'))
        torch_dtqd = torch.mm(torch.mm(torch.transpose(torch_d, 0, 1), torch_Q), torch_d).to(torch.device('cuda'))
        torch_alpha = - torch_gd / torch_dtqd
        torch_J = torch_J + torch_alpha * torch_d
        J = torch_J.to(torch.device('cpu')).numpy()
        # set all negatives to 0
        cond = np.where(J < 0, True, False)
        J[cond] = 0
        # end set
        torch_J = torch.from_numpy(J).to(torch.device('cuda'))
        torch_g = (2 * (torch.mm(torch_Q, torch_J) - torch_b)).to(torch.device('cuda'))
        g = torch_g.to(torch.device('cpu')).numpy()
        err = np.linalg(g)
        print(err)
        if err < 1e-4:
            return J
        torch_beta = torch.mm(torch.mm(torch.transpose(torch_g, 0, 1), torch_Q), torch_d) / torch_dtqd
        torch_d = (-torch_g + torch_beta * torch_d).to(torch.device('cuda'))

if __name__ == "__main__":
    source_seq(1, 0.005, 10)
    print(midpoint_algo(50))