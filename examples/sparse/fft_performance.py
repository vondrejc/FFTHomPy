from ffthompy.tensors import DFT, Tensor
import numpy as np
from ffthompy import Timer

import matplotlib.pyplot as plt

k_list = {'2': [1, 2, 3, 4, 5],
          '3': [1, 2, 3]}

for dim in [2]:
    for gridkind in [0, 1, 2]:
        if gridkind == 0:
            Ns = 5*np.power(3, k_list['{}'.format(dim)])

        elif gridkind == 1:
            Ns = 5*(np.power(2, np.add(k_list['{}'.format(dim)],
                                       4) - 1))  # 5*(np.power(2, np.add(k_list['{}'.format(dim)],2)-1))

        elif gridkind == 2:
            Ns = 3*np.power(3, np.add(k_list['{}'.format(dim)], 1))

        elif gridkind == 3:
            Ns = {'2': [15, 45, 75, 135, 155, 315, 405, 735, 875, 1125, 1215],  #
                  '3': [15, 45, 75, 105, 125, 135]}['{}'.format(dim)]

        time_list = [None]*len(Ns)

        for grid in range(len(Ns)):
            N = dim*(Ns[grid],)

            T = Tensor(name = 'EN', N = N, shape = (), Fourier = False)
            T.randomize()
            F = DFT(name = 'FN', inverse = False, N = N)  # discrete Fourier transform (DFT)
            iF = DFT(name = 'FiN', inverse = True, N = N)  # inverse DFT

            mean = 0

            for i in range(10):
                tic = Timer(name = 'FFT time')
                T = F(T)
                T = iF(T)
                tic.measure()
                mean = mean + tic.vals[0][0]

            time_list[grid] = mean/10

        plt.plot(Ns, time_list, 'x-', label = 'Grid kind {}'.format(gridkind), markevery = 1, )
    plt.legend()
plt.show()
