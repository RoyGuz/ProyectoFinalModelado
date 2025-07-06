import pandas as pd
import matplotlib.pyplot as plt

def graficar_tiempos(csv_path):

    df = pd.read_csv(csv_path)

    # Filtrar solo los casos con muestras válidas (evita errores si hubo n_validas = 0)
    df = df[(df["n_filtrado_sec"] > 0) & (df["n_filtrado_par"] > 0)]

    plt.figure(figsize=(8,5))
    plt.plot(df["n_muestras"], df["tiempo_sec_s"]/60, marker='o', label="Secuencial")
    plt.plot(df["n_muestras"], df["tiempo_par_s"]/60, marker='o', label="Paralelo")
    plt.xlabel("Cantidad de muestras generadas")
    plt.ylabel("Tiempo de ejecución [min]")
    plt.title("Tiempo de ejecución: Secuencial vs Paralelo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()