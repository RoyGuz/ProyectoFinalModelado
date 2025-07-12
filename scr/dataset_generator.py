from pathlib import Path
import sys

BASE_DIR = Path().resolve()
sys.path.append(str(BASE_DIR.parent / 'scr'))

from solver_fd import temp_chapa_P
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os


def variables(Nx, Ny):

    """
    Esta funcion genera condiciones de borde aleatorias (tanto de flujo como temperatura) y un punto caliente para
    la funcion "temp_chapa_P", seleccionando aleatoriamente un material desde la lista de materiales CSV y ajustando
    la T_max al punto de fusión del material seleccionado.
    
    Ingresa: 
        Nx (int): Cantidad de nodos de la placa en la direccion x
        Ny (int): Cantidad de nodos de la placa en la direccion y

    Retorna:
        cond_contor (dict): condiciones de borde listas para simular.
            Ej: cond_contor = {'A':1000,'B':15,'C':9000,'D':-10000}

        typ_cond_contorno (dict): tipo de condición de borde ('temp' o 'flu').
            Ej: typ_cond_contorno = {'A':'temp','B':'temp','C':'flu','D':'flu'}

        hot_point (dict): punto caliente.
            Ej: hot_point = {'i': 20, 'j': 15, 'T': 200}

        k (float): conductividad térmica del material.
        material_nombre (str): nombre del material utilizado.

    """
    #   Se busca el archivo CSV donde se tiene la lista de materiales
    materiales_path= BASE_DIR.parent / 'data' / 'materiales.csv'

    
    #   Se carga la lista de materiales
    df_materiales = pd.read_csv(materiales_path, sep=';')
    
    #   Se selecciona aleatoriamente un matrial 
    material = df_materiales.sample(1).iloc[0]      #   Funcion interna de pandas para seleccionar una fila de la lista cargada
    
    #   Se importan las caracteristicas utilizadas de fila seleccionada  
    material_nombre = material['Material']
    k = material['k [W/m·K]']
    T_fusion = material['Fusión [°C] (aprox)']
    
    #   Se define el rango de temperatura para el material en estudio
    T_min = -196
    T_max = T_fusion - 10

    #   Se define un rango generico para los flujos 
    q_min = -10000
    q_max = 10000

    #...................................................................................................................................
    #   Se defienen los diccionarios que van a contener mis datos
    cond_contor = {}
    typ_cond_contorno = {}
    bordes = ['A', 'B', 'C', 'D']

    for borde in bordes:
        tipo = np.random.choice(['temp', 'flu'])
        typ_cond_contorno[borde] = tipo

        if tipo == 'temp':
            valor = round(np.random.uniform(T_min, T_max),3)
        elif tipo == 'flu':
            valor = round(np.random.uniform(q_min, q_max),3)
        
        cond_contor[borde] = valor

    #   Se generan los datos del punto caliente
    i_hp = np.random.randint(0, Nx)
    j_hp = np.random.randint(0, Ny)
    T_hp = round(np.random.uniform(T_min, T_max),3)

    
    hot_point = {'i': i_hp, 'j': j_hp, 'T': T_hp}

    return cond_contor, typ_cond_contorno, hot_point, k, material_nombre,T_fusion

#..............................................................................................................................................................
#..............................................................................................................................................................

def generar_dataset(n_muestras, Nx, Ny, dx, dy, subfolder_name = None):
    """
    Esta funcion genera los datos necesarios para el entrenamiento de la Red Neuronal, toma un conjunto de datos
    de entrada, que se generan aleatoriamente con la funcion "variables", y los evalua en la funcion temp_chapa_P. 
    Los datos de salida junto con los de entrada se guardan en archivos .npy y tambien se genera un registro de todas 
    las combinaciones simuladas. 

    Ingresa: 
        n_muestras (int): Cantidad de muestras para simulacion.
        Nx (int): Cantidad de nodos de la placa en la direccion x
        Ny (int): Cantidad de nodos de la placa en la direccion y
        dx (float): Distancia entre nodos de la placa en la direccion x - [m]
        dy (float): Distancia entre nodos de la placa en la direccion y - [m]

    Retorna:
        No se retorna ninguna variable.
    
    Se generan 9 archivos:
        X (.npy): Datos de entrada para la Red Neuronal
        Y (.npy): Datos de salida para la Red Neuronal
        dataset_variables (.csv): Combinaciones simuladas

        X_train (.npy): Datos de entrenamiento de entrada para la Red Neuronal
        Y_train (.npy): Datos de entrenamiento de salida para la Red Neuronal
        dataset_variables_train (.csv): Combinaciones de entrenamiento simuladas 

        X_val (.npy): Datos de validacion de entrada para la Red Neuronal
        Y_val (.npy): Datos de validacion de salida para la Red Neuronal
        dataset_variables_val (.csv): Combinaciones de validacion simuladas 

    """
    #   Se definen las variables donde se almacenaran los datos para cada combinacion
    X = []
    Y = []
    registros = []

    #   Bucle principal
    for i in range(n_muestras):

        #   Genero mis variables aleatorias
        cond_contor, typ_cond_contorno, hot_point, k, material_nombre,T_fusion = variables(Nx, Ny)
        
        #......................... DATOS DE ENTRADA ......................................
        tipo_map = {'temp': 0, 'flu': 1}

        tipo_A = tipo_map[typ_cond_contorno['A']]
        tipo_B = tipo_map[typ_cond_contorno['B']]
        tipo_C = tipo_map[typ_cond_contorno['C']]
        tipo_D = tipo_map[typ_cond_contorno['D']]
        
        x_muestra = [
            
            k,
            hot_point['T'],
            hot_point['i'],
            hot_point['j'],
            tipo_A,
            tipo_B,
            tipo_C,
            tipo_D,
            cond_contor['A'],
            cond_contor['B'],
            cond_contor['C'],
            cond_contor['D']

        ]
        X.append(x_muestra)

        #.................... DATOS DE SALIDA .........................................
        #   Se calcula la distribucion de temperatura 
        T = temp_chapa_P(cond_contor, Nx, Ny, typ_cond_contorno, dx, dy, k, hot_point)

        y_muestra = T.flatten() 
        Y.append(y_muestra)
        
        #.................... REGISTROS DE CADA COMBINACION ...........................
        registros.append({

            'material': material_nombre,
            'k': k,
            'T_fusion' : T_fusion,
            'T_hp': hot_point['T'],
            'i_hp': hot_point['i'],
            'j_hp': hot_point['j'],
            'tipo_A': typ_cond_contorno['A'],
            'tipo_B': typ_cond_contorno['B'],
            'tipo_C': typ_cond_contorno['C'],
            'tipo_D': typ_cond_contorno['D'],
            'valor_A': cond_contor['A'],
            'valor_B': cond_contor['B'],
            'valor_C': cond_contor['C'],
            'valor_D': cond_contor['D']

        })

        if i % 50 == 0:
            print(f"Se tienen {i}/{n_muestras} muestras generadas.")

    #................................ ACONDICIONAMIENTO DE DATOS .........................................................
    X = np.array(X)
    Y = np.array(Y)    
    df_registros = pd.DataFrame(registros)# Para convertir el diccionario "registros" en una fila para guardar.

    T_min = Y.min(axis=1)
    T_max = Y.max(axis=1)
    T_fusion = df_registros['T_fusion'].values

    idx_validos = (T_min >= -250) & (T_max <= (T_fusion - 10))  #   Filtro Booleano

    X = X[idx_validos]
    Y = Y[idx_validos]
    df_registros = df_registros.iloc[idx_validos].reset_index(drop=True)

    print(f"\nFiltrado completado:")
    print(f"Muestras originales: {n_muestras}")
    print(f"Muestras después del filtrado: {len(Y)}")


    #................................ ALMACENAMIENTO DE DATOS .............................................................
    #   Defino la carpeta donde se guardaran los datos
    base_folder = BASE_DIR.parent / 'data' 

    if subfolder_name is not None:

        save_folder = os.path.join(base_folder, subfolder_name)#    Genero la ruta de guardado ... Ej: 'C:/Users/royer/Documents/ProyectoFinalModelado/data/dataset_5000_test/'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f'Se creo la carpeta: {save_folder}')
    else:
        save_folder = base_folder#  Esto es por si no se indico ninguna ruta

    # Separación reproducible 70% entrenamiento y 30% validación
    X_train, X_val, Y_train, Y_val, registros_train, registros_val = train_test_split(
        X, Y, df_registros, test_size=0.3, random_state=42, shuffle=True)
    
    # Guardado de archivos
    np.save(os.path.join(save_folder, 'X.npy'), X)
    np.save(os.path.join(save_folder, 'Y.npy'), Y)
    df_registros.to_csv(os.path.join(save_folder, 'dataset_variables.csv'), index=False, sep=';')

    # Guardar datos de entrenamiento
    np.save(os.path.join(save_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(save_folder, 'Y_train.npy'), Y_train)
    registros_train.to_csv(os.path.join(save_folder, 'dataset_variables_train.csv'), index=False, sep=';')

    # Guardar datos de validación
    np.save(os.path.join(save_folder, 'X_val.npy'), X_val)
    np.save(os.path.join(save_folder, 'Y_val.npy'), Y_val)
    registros_val.to_csv(os.path.join(save_folder, 'dataset_variables_val.csv'), index=False, sep=';')

    # print(f"Se genero el Dataset completo con {n_muestras} muestras y guardado en {save_folder}.")

    # print(f"Se genero la separacion del Dataset: entrenamiento (70%) y validación (30%)")

    return len(Y)

def variables_old():

    """
    Esta funcion genera condiciones de borde aleatorias (tanto de flujo como temperatura) y un punto caliente para
    la funcion "temp_chapa_P", seleccionando aleatoriamente un material desde la lista de materiales CSV y ajustando
    la T_max al punto de fusión del material seleccionado.
    
    Ingresa: 
        Nx (int): Cantidad de nodos de la placa en la direccion x
        Ny (int): Cantidad de nodos de la placa en la direccion y

    Retorna:
        cond_contor (dict): condiciones de borde listas para simular.
            Ej: cond_contor = {'A':1000,'B':15,'C':0,'D':0}

        typ_cond_contorno (dict): tipo de condición de borde ('temp' o 'flu').
            Ej: typ_cond_contorno = {'A':'temp','B':'temp','C':'flu','D':'flu'}

        k (float): conductividad térmica del material.
        material_nombre (str): nombre del material utilizado.

    """
    #   Se busca el archivo CSV donde se tiene la lista de materiales
    materiales_path= BASE_DIR.parent / 'data' / 'materiales.csv'

    
    #   Se carga la lista de materiales
    df_materiales = pd.read_csv(materiales_path, sep=';')

    while True:
        #   Se selecciona aleatoriamente un matrial 
        material = df_materiales.sample(1).iloc[0]      #   Funcion interna de pandas para seleccionar una fila de la lista cargada
        
        #   Se importan las caracteristicas utilizadas de fila seleccionada  
        material_nombre = material['Material']
        k = material['k [W/m·K]']
        T_fusion = material['Fusión [°C] (aprox)']
        if k>10:
            break
    
    #   Se define el rango de temperatura para el material en estudio
    T_min = -196
    T_max = T_fusion*0.4
    f_min = -2000
    f_max = 2000

    #...................................................................................................................................
    #   Se defienen los diccionarios que van a contener mis datos
 #   while True:
    cond_contor = {}
    typ_cond_contorno = {}
    bordes = ['A', 'B', 'C', 'D']

    for borde in bordes:
        tipo = np.random.choice(['temp', 'flu'])
        typ_cond_contorno[borde] = tipo

        if tipo == 'temp':
            valor = round(np.random.uniform(T_min, T_max),3)
        elif tipo == 'flu':
            valor = round(np.random.uniform(f_min, f_max),3)
            
        cond_contor[borde] = valor

        # # Verificar cantidad de bordes con flujo
        # count_flu = sum(1 for tipo in typ_cond_contorno.values() if tipo == 'flu')

        # if count_flu < 3:
        #     break

    return cond_contor, typ_cond_contorno, k, material_nombre,T_fusion

def generar_dataset_old(n_muestras, Nx, Ny, dx, dy, subfolder_name = None):
    """
    Esta funcion genera los datos necesarios para el entrenamiento de la Red Neuronal, toma un conjunto de datos
    de entrada, que se generan aleatoriamente con la funcion "variables", y los evalua en la funcion temp_chapa_P. 
    Los datos de salida junto con los de entrada se guardan en archivos .npy y tambien se genera un registro de todas 
    las combinaciones simuladas. 

    Ingresa: 
        n_muestras (int): Cantidad de muestras para simulacion.
        Nx (int): Cantidad de nodos de la placa en la direccion x
        Ny (int): Cantidad de nodos de la placa en la direccion y
        dx (float): Distancia entre nodos de la placa en la direccion x - [m]
        dy (float): Distancia entre nodos de la placa en la direccion y - [m]

    Retorna:
        No se retorna ninguna variable.
    
    Se generan 9 archivos:
        X (.npy): Datos de entrada para la Red Neuronal
        Y (.npy): Datos de salida para la Red Neuronal
        dataset_variables (.csv): Combinaciones simuladas

        X_train (.npy): Datos de entrenamiento de entrada para la Red Neuronal
        Y_train (.npy): Datos de entrenamiento de salida para la Red Neuronal
        dataset_variables_train (.csv): Combinaciones de entrenamiento simuladas 

        X_val (.npy): Datos de validacion de entrada para la Red Neuronal
        Y_val (.npy): Datos de validacion de salida para la Red Neuronal
        dataset_variables_val (.csv): Combinaciones de validacion simuladas 

    """
    #   Se definen las variables donde se almacenaran los datos para cada combinacion
    X = []
    Y = []
    registros = []

    #   Bucle principal
    for i in range(n_muestras):

        #   Genero mis variables aleatorias
        cond_contor, typ_cond_contorno, k, material_nombre,T_fusion = variables_old()
        
        #......................... DATOS DE ENTRADA ......................................
        tipo_map = {'temp': 0, 'flu': 1}

        tipo_A = tipo_map[typ_cond_contorno['A']]
        tipo_B = tipo_map[typ_cond_contorno['B']]
        tipo_C = tipo_map[typ_cond_contorno['C']]
        tipo_D = tipo_map[typ_cond_contorno['D']]
        
        x_muestra = [
            
            k,
            tipo_A,
            tipo_B,
            tipo_C,
            tipo_D,
            cond_contor['A'],
            cond_contor['B'],
            cond_contor['C'],
            cond_contor['D']

        ]
        X.append(x_muestra)

        #.................... DATOS DE SALIDA .........................................
        #   Se calcula la distribucion de temperatura 
        T = temp_chapa_P(cond_contor, Nx, Ny, typ_cond_contorno, dx, dy, k, hot_point=None)

        y_muestra = T.flatten() 
        Y.append(y_muestra)
        
        #.................... REGISTROS DE CADA COMBINACION ...........................
        registros.append({

            'material': material_nombre,
            'k': k,
            'T_fusion' : T_fusion,
            'tipo_A': typ_cond_contorno['A'],
            'tipo_B': typ_cond_contorno['B'],
            'tipo_C': typ_cond_contorno['C'],
            'tipo_D': typ_cond_contorno['D'],
            'valor_A': cond_contor['A'],
            'valor_B': cond_contor['B'],
            'valor_C': cond_contor['C'],
            'valor_D': cond_contor['D']

        })

        if i % 50 == 0:
            print(f"Se tienen {i}/{n_muestras} muestras generadas.")

    #................................ ACONDICIONAMIENTO DE DATOS .........................................................
    X = np.array(X)
    Y = np.array(Y)    
    df_registros = pd.DataFrame(registros)# Para convertir el diccionario "registros" en una fila para guardar.

    T_min = Y.min(axis=1)
    T_max = Y.max(axis=1)
    T_fusion = df_registros['T_fusion'].values

    idx_validos = (T_min >= -250) & (T_max <= (T_fusion*0.5))  #   Filtro Booleano

    X = X[idx_validos]
    Y = Y[idx_validos]
    df_registros = df_registros.iloc[idx_validos].reset_index(drop=True)

    print(f"\nFiltrado completado:")
    print(f"Muestras originales: {n_muestras}")
    print(f"Muestras después del filtrado: {len(Y)}")


    #................................ ALMACENAMIENTO DE DATOS .............................................................
    #   Defino la carpeta donde se guardaran los datos
    base_folder = BASE_DIR.parent / 'data' 

    if subfolder_name is not None:

        save_folder = os.path.join(base_folder, subfolder_name)#    Genero la ruta de guardado ... Ej: 'C:/Users/royer/Documents/ProyectoFinalModelado/data/dataset_5000_test/'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f'Se creo la carpeta: {save_folder}')
    else:
        save_folder = base_folder#  Esto es por si no se indico ninguna ruta

    # Separación reproducible 70% entrenamiento y 30% validación
    X_train, X_val, Y_train, Y_val, registros_train, registros_val = train_test_split(
        X, Y, df_registros, test_size=0.3, random_state=42, shuffle=True)
    
    # Guardado de archivos
    np.save(os.path.join(save_folder, 'X.npy'), X)
    np.save(os.path.join(save_folder, 'Y.npy'), Y)
    df_registros.to_csv(os.path.join(save_folder, 'dataset_variables.csv'), index=False, sep=';')

    # Guardar datos de entrenamiento
    np.save(os.path.join(save_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(save_folder, 'Y_train.npy'), Y_train)
    registros_train.to_csv(os.path.join(save_folder, 'dataset_variables_train.csv'), index=False, sep=';')

    # Guardar datos de validación
    np.save(os.path.join(save_folder, 'X_val.npy'), X_val)
    np.save(os.path.join(save_folder, 'Y_val.npy'), Y_val)
    registros_val.to_csv(os.path.join(save_folder, 'dataset_variables_val.csv'), index=False, sep=';')

    # print(f"Se genero el Dataset completo con {n_muestras} muestras y guardado en {save_folder}.")

    # print(f"Se genero la separacion del Dataset: entrenamiento (70%) y validación (30%)")

    return len(Y)