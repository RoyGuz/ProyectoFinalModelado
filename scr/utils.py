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

def comparar_T(T1, T2, Nx, Ny, etiquetas=('Versión 1', 'Versión 2')):
    """
    Compara dos distribuciones de temperatura T1 y T2:
    - Muestra las dos distribuciones lado a lado.
    - Muestra la diferencia absoluta entre ellas.
    - Calcula métricas de diferencia.

    Parámetros:
        T1, T2: arrays 1D o 2D de temperatura.
        Nx, Ny: dimensiones de la malla.
        etiquetas: tupla con etiquetas para las versiones.
    """

    # Asegura que tengan forma (Ny, Nx)
    T1 = np.array(T1).reshape((Ny, Nx))
    T2 = np.array(T2).reshape((Ny, Nx))
    diferencia = np.abs(T1 - T2)

    # Métricas de diferencia
    diff_max = np.max(diferencia)
    diff_mean = np.mean(diferencia)
    diff_std = np.std(diferencia)

    print(f"Comparación de distribuciones:")
    print(f"Diferencia máxima: {diff_max:.4f} °C")
    print(f"Diferencia media:  {diff_mean:.4f} °C")
    print(f"Desvío estándar:   {diff_std:.4f} °C")

    # Gráficos
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axs[0].imshow(T1, origin='lower', cmap='plasma')
    axs[0].set_title(f'{etiquetas[0]}')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(T2, origin='lower', cmap='plasma')
    axs[1].set_title(f'{etiquetas[1]}')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(diferencia, origin='lower', cmap='viridis')
    axs[2].set_title('Diferencia absoluta |T1 - T2| [°C]')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel('i (x)')
        ax.set_ylabel('j (y)')

    plt.suptitle('Comparación de distribuciones de temperatura')
    plt.tight_layout()
    plt.show()