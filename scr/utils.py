import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def graficarChapa(T,Nx,Ny):

    fig, ax = plt.subplots(figsize=(8, 6))

    numSeparaciones = 100  
    numLineas = 16     

    mappable = ax.contourf(T.reshape(Nx, Ny), levels=numSeparaciones, origin='upper', cmap='plasma')

    levels = ax.contour(T.reshape(Nx, Ny), levels=numLineas, colors='k', linewidths=1,origin='upper')

    cbar = plt.colorbar(mappable)
    cbar.set_label('T (°C)')

    ax.clabel(levels, inline=True, fontsize=10, fmt='%1.0f')


    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_title('Distribución de Temperatura para una placa cuadrada')

    plt.tight_layout()
    plt.show()

    return