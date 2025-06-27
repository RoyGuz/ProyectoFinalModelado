import numpy as np

def index(i, j,Nx):
    
    """
    Convierte coordenadas (i, j) a índice lineal para una malla Nx × Ny.
    """

    return j * Nx + i


def temp_chapa_P(cond_contor, Nx, Ny, typ_cond_contorno, dx, dy, k, hot_point):

    """
    Resuelve el problema de conducción estacionaria 2D en una placa mediante diferencias finitas.

    Parámetros:

        cond_contor (dict): Diccionario con condiciones de contorno ('A', 'B', 'C', 'D').
        Nx, Ny (int): Número de nodos en x e y.
        typ_cond_contorno (dict): Tipo de condición para cada borde: 'temp' o 'flu'.
        dx, dy (float): Tamaños de paso en x e y.
        k (float): Conductividad térmica.
        hot_point (dict or None): Punto caliente, con claves 'i', 'j', 'T'.

    Retorna:
        T: Matriz (Ny, Nx) con la distribución de temperatura.
    """
    
    beta = dx / dy
    N = Nx * Ny
    b = np.zeros(N)
    A = np.eye(N)

    for j in range(Ny):
        for i in range(Nx):
            
            idx = index(i, j,Nx)

            if hot_point is not None and i == hot_point['i'] and j == hot_point['j']:
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = hot_point['T']
                continue

            # BORDES
            if j == 0:  # Borde superior A
                if typ_cond_contorno['A'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = cond_contor['A']
                elif typ_cond_contorno['A'] == 'flu' and j+1 < Ny:
                    A[idx, :] = 0
                    A[idx, idx] = -1
                    A[idx, index(i, j+1,Nx)] = 1
                    b[idx] = dx * cond_contor['A'] / k

            elif j == Ny-1:  # Borde inferior C
                if typ_cond_contorno['C'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = cond_contor['C']
                elif typ_cond_contorno['C'] == 'flu' and j-1 >= 0:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    A[idx, index(i, j-1,Nx)] = -1
                    b[idx] = dx * cond_contor['C'] / k

            elif i == 0:  # Borde izquierdo D
                if typ_cond_contorno['D'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = cond_contor['D']
                elif typ_cond_contorno['D'] == 'flu' and i+1 < Nx:
                    A[idx, :] = 0
                    A[idx, idx] = -1
                    A[idx, index(i+1, j,Nx)] = 1
                    b[idx] = dy * cond_contor['D'] / k

            elif i == Nx-1:  # Borde derecho B
                if typ_cond_contorno['B'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = cond_contor['B']
                elif typ_cond_contorno['B'] == 'flu' and i-1 >= 0:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    A[idx, index(i-1, j,Nx)] = -1
                    b[idx] = dy * cond_contor['B'] / k

            else:  # NODOS INTERNOS
                A[idx, idx] = -2 * (1 + beta**2)
                A[idx, index(i-1, j,Nx)] = 1
                A[idx, index(i+1, j,Nx)] = 1
                A[idx, index(i, j-1,Nx)] = beta**2
                A[idx, index(i, j+1,Nx)] = beta**2

    T = np.linalg.solve(A, b)
    T = T.reshape((Ny, Nx))

    return T
