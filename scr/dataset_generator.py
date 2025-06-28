import pandas as pd
import numpy as np

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
    materiales_path='C:/Users/royer/Documents/ProyectoFinalModelado/data/materiales.csv'
    
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
            valor = np.random.uniform(T_min, T_max)
        elif tipo == 'flu':
            valor = np.random.uniform(q_min, q_max)
        
        cond_contor[borde] = valor

    #   Se generan los datos del punto caliente
    i_hp = np.random.randint(0, Nx)
    j_hp = np.random.randint(0, Ny)
    T_hp = np.random.uniform(T_min, T_max)
    
    hot_point = {'i': i_hp, 'j': j_hp, 'T': T_hp}

    return cond_contor, typ_cond_contorno, hot_point, k, material_nombre

