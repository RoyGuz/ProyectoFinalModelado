import numpy as np

def index(i, j,Nx):
    
    """
    Convierte coordenadas (i, j) a índice lineal para una malla Nx × Ny.
    """

    return j * Nx + i


def temp_chapa_P(cond_contor, Nx, Ny, typ_cond_contorno, dx, dy, k, hot_point):

    """
    Esta funcion resuelve el problema de conducción estacionaria 2D en una placa mediante diferencias finitas.

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
            
            idx = index(i, j,Nx)#   Indice 1D asociado el punto (i,j)

            if hot_point is not None and i == hot_point['i'] and j == hot_point['j']:
                A[idx, :] = 0
                A[idx, idx] = 1
                b[idx] = hot_point['T']
                continue

            #................ ESQUINAS: TEMP-TEMP y FLUJO-FLUJO ..............................

            esquina = False

            if i == 0 and j == 0:
                if typ_cond_contorno['A'] == 'temp' and typ_cond_contorno['D'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = 0.5 * (cond_contor['A'] + cond_contor['D'])
                    esquina = True
                elif typ_cond_contorno['A'] == 'flu' and typ_cond_contorno['D'] == 'flu':
                    A[idx, :] = 0
                    A[idx, idx] = -2
                    A[idx, index(i+1, j, Nx)] = 1
                    A[idx, index(i, j+1, Nx)] = 1
                    b[idx] = (dy * cond_contor['A'] + dx * cond_contor['D']) / k
                    esquina = True

            if i == Nx-1 and j == 0:
                if typ_cond_contorno['A'] == 'temp' and typ_cond_contorno['B'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = 0.5 * (cond_contor['A'] + cond_contor['B'])
                    esquina = True
                elif typ_cond_contorno['A'] == 'flu' and typ_cond_contorno['B'] == 'flu':
                    A[idx, :] = 0
                    A[idx, idx] = -2
                    A[idx, index(i-1, j, Nx)] = 1
                    A[idx, index(i, j+1, Nx)] = 1
                    b[idx] = (dy * cond_contor['A'] + dx * cond_contor['B']) / k
                    esquina = True

            if i == Nx-1 and j == Ny-1:
                if typ_cond_contorno['C'] == 'temp' and typ_cond_contorno['B'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = 0.5 * (cond_contor['C'] + cond_contor['B'])
                    esquina = True
                elif typ_cond_contorno['C'] == 'flu' and typ_cond_contorno['B'] == 'flu':
                    A[idx, :] = 0
                    A[idx, idx] = -2
                    A[idx, index(i-1, j, Nx)] = 1
                    A[idx, index(i, j-1, Nx)] = 1
                    b[idx] = (dy * cond_contor['C'] + dx * cond_contor['B']) / k
                    esquina = True

            if i == 0 and j == Ny-1:
                if typ_cond_contorno['C'] == 'temp' and typ_cond_contorno['D'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = 0.5 * (cond_contor['C'] + cond_contor['D'])
                    esquina = True
                elif typ_cond_contorno['C'] == 'flu' and typ_cond_contorno['D'] == 'flu':
                    A[idx, :] = 0
                    A[idx, idx] = -2
                    A[idx, index(i+1, j, Nx)] = 1
                    A[idx, index(i, j-1, Nx)] = 1
                    b[idx] = (dy * cond_contor['C'] + dx * cond_contor['D']) / k
                    esquina = True

            if esquina:
                continue  # Evitar procesar nuevamente el nodo           
           
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
                    b[idx] = dy * cond_contor['A'] / k

            elif j == Ny-1:  # Borde inferior C
                if typ_cond_contorno['C'] == 'temp':
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    b[idx] = cond_contor['C']
                elif typ_cond_contorno['C'] == 'flu' and j-1 >= 0:
                    A[idx, :] = 0
                    A[idx, idx] = 1
                    A[idx, index(i, j-1,Nx)] = -1
                    b[idx] = dy * cond_contor['C'] / k

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

#.....................................................................................................................................
#.....................................................................................................................................

def temp_chapa_P2(cond_contor, Nx, Ny, typ_cond_contorno, dx, dy, k, hot_point):

    """
    Esta funcion resuelve el problema de conducción estacionaria 2D en una placa mediante diferencias finitas 
    (VERSION ORIGINAL RESUELTA EN CLASE).

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
    
    beta = Nx / Ny

    # ...................... ARMADO DE LA MATRIZ A ..............................................................

    filas_internas = np.arange(1, Ny - 1)
    columnas_internas = np.arange(1, Nx - 1)
    indices_filas_int, indices_columnas_int = np.meshgrid(filas_internas, columnas_internas, indexing='ij')
    flat_indices = np.ravel_multi_index((indices_filas_int.flatten(), indices_columnas_int.flatten()), (Nx, Ny))

    A = np.eye(Nx * Ny)
    A[flat_indices, flat_indices] = -2 * (1 + beta ** 2)
    A[flat_indices, flat_indices - 1] = 1
    A[flat_indices, flat_indices + 1] = 1
    A[flat_indices, flat_indices - Nx] = beta ** 2
    A[flat_indices, flat_indices + Nx] = beta ** 2

    b = np.zeros((Nx, Ny))

    # ................. BORDE A / BORDE D .........................................................

    if typ_cond_contorno['A'] == 'temp' and typ_cond_contorno['D'] == 'flu':
        b[0, 0] = cond_contor['A']
    elif typ_cond_contorno['A'] == 'flu' and typ_cond_contorno['D'] == 'temp':
        b[0, 0] = cond_contor['D']
    elif typ_cond_contorno['A'] == 'temp' and typ_cond_contorno['D'] == 'temp':
        b[0, 0] = (cond_contor['A'] + cond_contor['D']) / 2

    if typ_cond_contorno['A'] == 'temp':
        b[0, 1:Nx - 1] = cond_contor['A']
    elif typ_cond_contorno['A'] == 'flu':
        b[0, 1:Nx - 1] = dy * cond_contor['A'] / k

    # ................. BORDE A / BORDE B .........................................................

    if typ_cond_contorno['A'] == 'temp' and typ_cond_contorno['B'] == 'flu':
        b[0, Nx - 1] = cond_contor['A']
    elif typ_cond_contorno['A'] == 'flu' and typ_cond_contorno['B'] == 'temp':
        b[0, Nx - 1] = cond_contor['B']
    elif typ_cond_contorno['A'] == 'temp' and typ_cond_contorno['B'] == 'temp':
        b[0, Nx - 1] = (cond_contor['A'] + cond_contor['B']) / 2

    if typ_cond_contorno['B'] == 'temp':
        b[1:Ny - 1, -1] = cond_contor['B']
    elif typ_cond_contorno['B'] == 'flu':
        b[1:Ny - 1, -1] = dx * cond_contor['B'] / k

    # ................. BORDE B / BORDE C .........................................................
    
    if typ_cond_contorno['B'] == 'temp' and typ_cond_contorno['C'] == 'flu':
        b[-1, -1] = cond_contor['B']
    elif typ_cond_contorno['B'] == 'flu' and typ_cond_contorno['C'] == 'temp':
        b[-1, -1] = cond_contor['C']
    elif typ_cond_contorno['B'] == 'temp' and typ_cond_contorno['C'] == 'temp':
        b[-1, -1] = (cond_contor['B'] + cond_contor['C']) / 2

    if typ_cond_contorno['C'] == 'temp':
        b[-1, 1:Nx - 1] = cond_contor['C']
    elif typ_cond_contorno['C'] == 'flu':
        b[-1, 1:Nx - 1] = dy * cond_contor['C'] / k

    # ................. BORDE C / BORDE D .........................................................
    
    if typ_cond_contorno['C'] == 'temp' and typ_cond_contorno['D'] == 'flu':
        b[-1, 0] = cond_contor['C']
    elif typ_cond_contorno['C'] == 'flu' and typ_cond_contorno['D'] == 'temp':
        b[-1, 0] = cond_contor['D']
    elif typ_cond_contorno['C'] == 'temp' and typ_cond_contorno['D'] == 'temp':
        b[-1, 0] = (cond_contor['C'] + cond_contor['D']) / 2

    if typ_cond_contorno['D'] == 'temp':
        b[1:Ny - 1, 0] = cond_contor['D']
    elif typ_cond_contorno['D'] == 'flu':
        b[1:Ny - 1, 0] = dx * cond_contor['D'] / k
   
    b = b.flatten()

    if typ_cond_contorno['A'] == 'flu' and typ_cond_contorno['D'] == 'flu':
        idx = np.ravel_multi_index((0, 0), (Ny, Nx))
        A[idx, :] = 0
        A[idx, idx] = -2
        A[idx, idx + 1] = 1
        A[idx, idx + Nx] = 1
        b[idx] = (dy * cond_contor['A'] + dx * cond_contor['D']) / k

    if typ_cond_contorno['A'] == 'flu' and typ_cond_contorno['B'] == 'flu':
        idx = np.ravel_multi_index((0, Nx - 1), (Ny, Nx))
        A[idx, :] = 0
        A[idx, idx] = -2
        A[idx, idx - 1] = 1
        A[idx, idx + Nx] = 1
        b[idx] = (dy * cond_contor['A'] + dx * cond_contor['B']) / k

    if typ_cond_contorno['C'] == 'flu' and typ_cond_contorno['B'] == 'flu':
        idx = np.ravel_multi_index((Ny - 1, Nx - 1), (Ny, Nx))
        A[idx, :] = 0
        A[idx, idx] = -2
        A[idx, idx - 1] = 1
        A[idx, idx - Nx] = 1
        b[idx] = (dy * cond_contor['C'] + dx * cond_contor['B']) / k

    if typ_cond_contorno['C'] == 'flu' and typ_cond_contorno['D'] == 'flu':
        idx = np.ravel_multi_index((Ny - 1, 0), (Ny, Nx))
        A[idx, :] = 0
        A[idx, idx] = -2
        A[idx, idx + 1] = 1
        A[idx, idx - Nx] = 1
        b[idx] = (dy * cond_contor['C'] + dx * cond_contor['D']) / k

    # ............... AJUSTES DE FLUJO (q != 0) ..................................................................

    if typ_cond_contorno['A'] == 'flu':
        filaFlujo = np.zeros(Nx - 2, dtype=int)                   # j = 0
        columnaFlujo = np.arange(1, Nx - 1, dtype=int)
        indices_filas_int, indices_columnas_int = np.meshgrid(filaFlujo, columnaFlujo, indexing='ij')
        indices = np.ravel_multi_index((indices_filas_int.flatten(), indices_columnas_int.flatten()), (Ny, Nx))

        A[indices, :] = 0
        A[indices, indices] = -1
        A[indices, indices + Nx] = 1
        b[indices] = dy * cond_contor['A'] / k

    if typ_cond_contorno['C'] == 'flu':
        filaFlujo = np.full(Nx - 2, Ny - 1, dtype=int)            # j = Ny - 1
        columnaFlujo = np.arange(1, Nx - 1, dtype=int)
        indices_filas_int, indices_columnas_int = np.meshgrid(filaFlujo, columnaFlujo, indexing='ij')
        indices = np.ravel_multi_index((indices_filas_int.flatten(), indices_columnas_int.flatten()), (Ny, Nx))

        A[indices, :] = 0
        A[indices, indices] = 1
        A[indices, indices - Nx] = -1
        b[indices] = dy * cond_contor['C'] / k

    if typ_cond_contorno['D'] == 'flu':
        filaFlujo = np.arange(1, Ny - 1, dtype=int)
        columnaFlujo = np.zeros(Ny - 2, dtype=int)
        indices_filas_int, indices_columnas_int = np.meshgrid(filaFlujo, columnaFlujo, indexing='ij')
        indices = np.ravel_multi_index((indices_filas_int.flatten(), indices_columnas_int.flatten()), (Ny, Nx))

        A[indices, :] = 0
        A[indices, indices] = -1
        A[indices, indices + 1] = 1
        b[indices] = dx * cond_contor['D'] / k

    if typ_cond_contorno['B'] == 'flu':
        filaFlujo = np.arange(1, Ny - 1, dtype=int)
        columnaFlujo = np.full(Ny - 2, Nx - 1, dtype=int)
        indices_filas_int, indices_columnas_int = np.meshgrid(filaFlujo, columnaFlujo, indexing='ij')
        indices = np.ravel_multi_index((indices_filas_int.flatten(), indices_columnas_int.flatten()), (Ny, Nx))

        A[indices, :] = 0
        A[indices, indices] = 1
        A[indices, indices - 1] = -1
        b[indices] = dx * cond_contor['B'] / k


    #.................... AJUSTES DEL PUNTO CALIENTE ..................................................................

    if hot_point is not None:
        i_hp = hot_point['i']
        j_hp = hot_point['j']
        T_hp = hot_point['T']
        idx_hp = np.ravel_multi_index((j_hp, i_hp), (Nx, Ny))
        A[idx_hp, :] = 0
        A[idx_hp, idx_hp] = 1
        b[idx_hp] = T_hp

    T = np.linalg.solve(A, b)

    return T
