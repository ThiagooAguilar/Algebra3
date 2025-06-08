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
    n = len(A)
    A_copy = np.copy(A) # Para no modificar la matriz dada
    b_copy = np.copy(b)

    for k in range(n-1): 
        factor = A_copy[k+1][k] / A_copy[k][k] 
        A_copy[k+1][k] = 0 # La subdiagonal -> a = 0 siempre
        A_copy[k+1][k+1] -= factor * A_copy[k][k+1]
        # La superdiagonal(c) queda igual
        b_copy[k+1] -= factor * b_copy[k]

    x = [0] * n
    x[-1] = b_copy[-1] / A_copy[-1][-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b_copy[i] - A_copy[i][i + 1] * x[i + 1]) / A_copy[i][i]

    return x

def gauss_pivoteo(A, b):
    n = len(A)
    A_copy = np.copy(A) # Para no modificar la matriz dada
    b_copy = np.copy(b)

    for k in range(n - 1): 
        max_i = k
        for i in range(k+1, n):
            if abs(A_copy[i][k]) > abs(A_copy[max_i][k]):
                max_i = i
        if max_i != k: 
            A_copy[[k, max_i]] = A_copy[[max_i, k]]

            b_copy[k], b_copy[max_i] = b_copy[max_i], b_copy[k]
        for j in range(k+1,n): # Para Normalizar el pivote
            A_copy[k][j] /= A_copy[k][k]
        b_copy[k] /= A_copy[k][k]
        A_copy[k][k] = 1

        for i in range(k+1,n):
            factor = A_copy[i][k]  # A[k][k] =1
            for j in range(k+1, n):
                A_copy[i][j] -= factor * A_copy[k][j] 
            b_copy[i] -= factor * b_copy[k]
            A_copy[i][k] = 0


    x = [0] * n
    x[-1] = b_copy[-1] / A_copy[-1][-1]
    for i in range(n - 2, -1, -1):
        sumatoria = 0
        for k in range(i+1, n):
            sumatoria += A_copy[i][k] * x[k]
        x[i] = (b[i] - sumatoria) / A_copy[i][i]

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

resoluciones = [20, 30, 50, 70, 100, 500, 1000]
dt = 0.1
alpha = 0.01
pasos = 10
metodos = ['directo', 'optimizado', 'gauss_pivoteo']

# TODO: correr experimientos, obtener resultados y compararlos
