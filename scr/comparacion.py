import time
import csv
from dataset_generator_P import generar_dataset_paralelo
from dataset_generator import generar_dataset

def comparar_tiempos(n_muestras_list, Nx, Ny, dx, dy):
    tiempos_sec = []
    tiempos_par = []
    speedups = []
    n_filtrado_sec = []
    n_filtrado_par = []

    for n_muestras in n_muestras_list:
        print(f"\n=== Probando con {n_muestras} muestras ===")

        start_time = time.time()
        n_validas_sec = generar_dataset(n_muestras, Nx, Ny, dx, dy, subfolder_name=f"test_secuencial_{n_muestras}")
        sec_time = time.time() - start_time
        tiempos_sec.append(sec_time)
        n_filtrado_sec.append(n_validas_sec)
        print(f"Tiempo secuencial: {sec_time:.2f} s")
        print(f"Muestras válidas post-filtrado (sec): {n_validas_sec}")

        start_time = time.time()
        n_validas_par = generar_dataset_paralelo(n_muestras, Nx, Ny, dx, dy, subfolder_name=f"test_paralelo_{n_muestras}")
        par_time = time.time() - start_time
        tiempos_par.append(par_time)
        n_filtrado_par.append(n_validas_par)
        print(f"Tiempo paralelo: {par_time:.2f} s")
        print(f"Muestras válidas post-filtrado (par): {n_validas_par}")

        speedup = sec_time / par_time
        speedups.append(speedup)
        print(f"Speedup: {speedup:.2f}x más rápido en paralelo")

    with open("benchmark_tiempos.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "n_muestras",
            "tiempo_sec_s",
            "tiempo_par_s",
            "speedup",
            "n_filtrado_sec",
            "n_filtrado_par"
        ])
        for n, t_sec, t_par, sp, nf_sec, nf_par in zip(
            n_muestras_list, tiempos_sec, tiempos_par, speedups, n_filtrado_sec, n_filtrado_par
        ):
            writer.writerow([n, t_sec, t_par, sp, nf_sec, nf_par])

    print("\nResultados guardados en 'benchmark_tiempos.csv'.")

if __name__ == "__main__":
    comparar_tiempos(
        n_muestras_list=[10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        Nx=50,
        Ny=50,
        dx=0.05,
        dy=0.05
    )
