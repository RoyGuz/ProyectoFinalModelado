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

#....................................................................................................................

from tabulate import tabulate

def mostrar_tabla_variables_ordenada(cond_contor, typ_cond_contorno, hot_point, material_nombre):

    print("\n=== Condiciones de Contorno ===")

    tabla_contorno = []
    for borde in cond_contor.keys():
        tipo = typ_cond_contorno.get(borde, 'No definido')
        valor = cond_contor[borde]
        tabla_contorno.append([borde, tipo, valor])

    print(tabulate(tabla_contorno, headers=["Borde", "Tipo de condición", "Valor"], tablefmt="grid"))

    print("\n=== Punto Caliente ===")

    tabla_hot = []

    tabla_hot = [[hot_point['i'], hot_point['j'], hot_point['T']]]
    print(tabulate(tabla_hot, headers=["i", "j", "Temp"], tablefmt="grid"))

    print(f"\nMaterial: {material_nombre}")