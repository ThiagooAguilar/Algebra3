# TP Simulación Física en Videojuegos: Difusión de Calor 2D

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

# Construcción matriz A para método implícito
def construir_matriz(nx, ny, dx, dy, dt, alpha):
    Nix, Niy = nx - 2, ny - 2
    Ix = identity(Nix)
    Iy = identity(Niy)

    main_diag_x = 2 * (1/dx**2 + 1/dy**2) * np.ones(Nix)
    off_diag_x = -1/dx**2 * np.ones(Nix - 1)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1])

    off_diag_y = -1/dy**2 * np.ones(Niy - 1)
    Ty = diags([off_diag_y, off_diag_y], [-1, 1], shape=(Niy, Niy))

    L = kron(Iy, Tx) + kron(Ty, Ix)
    A = identity(Nix*Niy) - dt * alpha * L
    return A

# Inicializar temperatura y condiciones de frontera
def inicializar_T(nx, ny):
    T = np.ones((ny, nx)) * 25
    T[:, 0] = 100    # borde izquierdo
    T[:, -1] = 50    # borde derecho
    T[0, :] = 0      # borde superior
    T[-1, :] = 75    # borde inferior
    # Fuente interna caliente
    T[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T

# TODO: Métodos a desarrollar
def optimizado(A, b):
    """Resolución de un sistema de ecuaciones lineales optimizado
        cuando la matriz A es tridiagonal.
    """
    A = A.toarray()
    n = A.shape[0]
    A_copy = np.copy(A)
    b_copy = np.copy(b)

    for k in range(n-1):
        factor = A_copy[k+1][k] / A_copy[k][k]
        A_copy[k+1][k] = 0 # La subdiagonal -> a = 0 siempre
        A_copy[k+1][k+1] -= factor * A_copy[k][k+1]
        # La superdiagonal(c) queda igual
        b_copy[k+1] -= factor * b_copy[k]

    x = np.zeros(n)
    x[-1] = b_copy[-1] / A_copy[-1][-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b_copy[i] - A_copy[i][i + 1] * x[i + 1]) / A_copy[i][i]

    return x
def es_tridiagonal(A):
    A = np.asarray(A)
    n, m = A.shape
    if n != m:
        return False  # No es cuadrada
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i, j]) > 0:
                return False
    return True

def gauss_pivoteo(A, b):
    # Convertimos A a densa solo si no lo es
    if not isinstance(A, np.ndarray):
        AA = A.toarray()
    else:
        AA = A.copy()
    bb = b.copy()
    n = AA.shape[0]

    for k in range(n - 1):

        max_i = np.argmax(np.abs(AA[k:n, k])) + k # Busca el pivote con mayor modulo
        if max_i != k:
            AA[[k, max_i]] = AA[[max_i, k]]
            bb[k], bb[max_i] = bb[max_i], bb[k]

        AA[k, k + 1: ] /= AA[k, k] # Normaliza la diagonal a unos
        bb[k] /= AA[k, k]
        AA[k, k] = 1.0

        for i in range(k + 1, n): # Genera ceros
            factor = AA[i, k]
            AA[i, k + 1:] -= factor * AA[k, k + 1:]
            bb[i] -= factor * bb[k]
            AA[i, k] = 0.0

    x = np.zeros(n) # Sustitución hacia atrás
    x[-1] = bb[-1] / AA[-1, -1]
    for i in range(n - 2, -1, -1):
        x[i] = (bb[i] - np.dot(AA[i, i + 1:], x[i + 1:])) / AA[i, i]

    return x

# Un paso de simulación con el método implícito y método de solución
def paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion):
    b = T[1:-1, 1:-1].copy()
    # Incorporar condiciones de borde en b
    b[:, 0] += dt * alpha * T[1:-1, 0] / dx**2
    b[:, -1] += dt * alpha * T[1:-1, -1] / dx**2
    b[0, :] += dt * alpha * T[0, 1:-1] / dy**2
    b[-1, :] += dt * alpha * T[-1, 1:-1] / dy**2
    b = b.flatten()

    if metodo_solucion == 'directo':
        T_vec = spsolve(A, b)
    elif metodo_solucion == 'optimizado':
        T_vec = optimizado(A, b)
    elif metodo_solucion == 'gauss_pivoteo':
        T_vec  = gauss_pivoteo(A, b)
    else:
        raise ValueError("Método de solución no reconocido")

    T_new = T.copy()
    T_new[1:-1, 1:-1] = T_vec.reshape((ny - 2, nx - 2))
    # Mantener la fuente de calor interna fija
    T_new[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T_new

# Simular múltiples pasos, medir tiempos
def simular(nx, ny, dt, alpha, pasos, metodo_solucion):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    A = construir_matriz(nx, ny, dx, dy, dt, alpha)
    T = inicializar_T(nx, ny)

    tiempos = []
    for _ in range(pasos):
        start = time.time()
        T = paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion)
        end = time.time()
        tiempos.append(end - start)

    tiempo_promedio = np.mean(tiempos)
    return T, tiempo_promedio

# Error RMS relativo
def error_rms(T_ref, T):
    return np.sqrt(np.mean((T_ref - T)**2)) / np.sqrt(np.mean(T_ref**2))

# --------- Experimentos ---------

dt = 0.1
alpha = 0.01
pasos = 10
resoluciones = [10, 20, 30, 50]
metodos = ['directo', 'optimizado', 'gauss_pivoteo']

# --- Para almacenar resultados ---
tiempos = {m: [] for m in metodos}
errores = {m: [] for m in metodos if m != 'directo'}
tam_sistema = [((n - 2) ** 2)-9 for n in resoluciones]

print("=== CORRIENDO EXPERIMENTOS ===")
for res in resoluciones:
    print(f"\nResolución: {res}x{res} → sistema de {((res - 2) ** 2) - 9} incógnitas")

    # Método de referencia
    T_ref, t_directo = simular(res, res, dt, alpha, pasos, 'directo')
    tiempos['directo'].append(t_directo)
    print(f"  directo         → tiempo = {t_directo:.4f} s")

    for metodo in ['optimizado', 'gauss_pivoteo']:
        try:
            T_metodo, t_metodo = simular(res, res, dt, alpha, pasos, metodo)
            err = error_rms(T_ref, T_metodo)
            tiempos[metodo].append(t_metodo)
            errores[metodo].append(err)
            print(f"  {metodo:15} → tiempo = {t_metodo:.4f} s | error RMS = {err:.2e}")
        except Exception as e:
            print(f"  {metodo:15} → ERROR: {e}")
            tiempos[metodo].append(None)
            errores[metodo].append(None)

# --- Gráfico: Tiempo promedio vs Tamaño del sistema ---
plt.figure(figsize=(8, 5))
for metodo in metodos:
    plt.plot(tam_sistema, tiempos[metodo], marker='o', label=metodo)
plt.xlabel("Tamaño del sistema (N = ((n - 2)^2) - 9)")
plt.ylabel("Tiempo promedio por paso (s)")
plt.title("Tiempo vs Tamaño del sistema")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico: Error RMS vs Tamaño del sistema ---
plt.figure(figsize=(8, 5))
for metodo in errores:
    plt.plot(tam_sistema, errores[metodo], marker='o', label=metodo)
plt.xlabel("Tamaño del sistema (N = ((n - 2)^2) - 9)")
plt.ylabel("Error RMS relativo")
plt.title("Error numérico vs Tamaño del sistema")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico solo del error de Gauss vs Directo ---
errores_gauss = errores['gauss_pivoteo']

plt.figure(figsize=(8, 5))
plt.plot(tam_sistema, errores_gauss, marker='o', color='darkgreen', label='gauss_pivoteo')
plt.xlabel("Tamaño del sistema (N = (res - 2)^2 - 9)")
plt.ylabel("Error RMS relativo")
plt.title("Error numérico de gauss_pivoteo vs método directo")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# --- Visual de las temperaturas ---
resoluciones = [(10,10),(20,20),(30,30),(50,50)]
pasos = [1, 5, 10]
for nx, ny in resoluciones:
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    A = construir_matriz(nx, ny, dx, dy, dt, alpha)

    for metodo in metodos:
        T = inicializar_T(nx, ny)
        T_actual = T.copy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for paso in range(1, max(pasos) + 1):
            if paso != 0:
                T_actual = paso_simulacion(T_actual, A, nx, ny, dx, dy, dt, alpha, metodo)
            if paso in pasos:
                idx = pasos.index(paso)
                im = axes[idx].imshow(T_actual, cmap='hot', origin='lower')
                axes[idx].set_title(f"Paso {paso}")
                axes[idx].set_xlabel("x")
                axes[idx].set_ylabel("y")
                plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.suptitle(f"Evolución térmica - {nx}x{ny} - Método: {metodo}")
        plt.tight_layout()
        plt.show()