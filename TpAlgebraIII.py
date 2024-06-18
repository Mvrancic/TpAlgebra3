import numpy as np
import time


# Funciones de eliminación Gaussiana
def gauss_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        Ab[i] = Ab[i] / Ab[i][i]
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[i] * Ab[j][i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i][-1] - np.sum(Ab[i][i + 1:n] * x[i + 1:n])

    return x


def gauss_elimination_partial_pivoting(A, b):
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        Ab[i] = Ab[i] / Ab[i][i]
        for j in range(i + 1, n):
            Ab[j] = Ab[j] - Ab[i] * Ab[j][i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i][-1] - np.sum(Ab[i][i + 1:n] * x[i + 1:n])

    return x


def gauss_tridiagonal(A, b):
    n = len(b)
    a = np.zeros(n - 1)
    b_diag = np.zeros(n)
    c = np.zeros(n - 1)

    for i in range(n):
        b_diag[i] = A[i, i]
        if i < n - 1:
            c[i] = A[i, i + 1]
        if i > 0:
            a[i - 1] = A[i, i - 1]

    for i in range(1, n):
        m = a[i - 1] / b_diag[i - 1]
        b_diag[i] = b_diag[i] - m * c[i - 1]
        b[i] = b[i] - m * b[i - 1]

    x = np.zeros(n)
    x[-1] = b[-1] / b_diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / b_diag[i]

    return x


# Función para generar matrices tridiagonales
def generate_tridiagonal_matrix(n):
    A = np.zeros((n, n))
    b = np.random.rand(n)
    for i in range(n):
        A[i, i] = 2
        if i > 0:
            A[i, i - 1] = -1
        if i < n - 1:
            A[i, i + 1] = -1
    return A, b


# Función para medir el tiempo de ejecución
def measure_time(func, A, b):
    start_time = time.time()
    solution = func(A, b)
    end_time = time.time()
    return end_time - start_time, solution


# Función para calcular errores
def calculate_errors(x_true, x_approx):
    error_abs = np.linalg.norm(x_true - x_approx)
    error_rel = error_abs / np.linalg.norm(x_true)
    return error_abs, error_rel


# Crear y ejecutar casos de prueba
sizes = [30, 50]
tri_sizes = [100, 1000]
results_time = {}
results_precision = {}

# Pruebas para matrices tridiagonales
for size in tri_sizes:
    A, b = generate_tridiagonal_matrix(size)

    # Solución exacta usando numpy
    x_true = np.linalg.solve(A, b)

    # Medir tiempo para Gauss tridiagonal
    time_tri, sol_tri = measure_time(gauss_tridiagonal, A, b)

    # Medir tiempo para Gauss estándar
    time_standard, sol_standard = measure_time(gauss_elimination, A, b)

    # Calcular errores
    error_tri_abs, error_tri_rel = calculate_errors(x_true, sol_tri)
    error_standard_abs, error_standard_rel = calculate_errors(x_true, sol_standard)

    results_time[f"Tridiagonal {size}x{size}"] = {
        "Optimizado": time_tri,
        "No optimizado": time_standard
    }
    results_precision[f"Tridiagonal {size}x{size}"] = {
        "Error absoluto (optimizado)": error_tri_abs,
        "Error relativo (optimizado)": error_tri_rel,
        "Error absoluto (no optimizado)": error_standard_abs,
        "Error relativo (no optimizado)": error_standard_rel
    }

# Pruebas para matrices completas
for size in sizes:
    A = np.random.rand(size, size)
    b = np.random.rand(size)

    # Solución exacta usando numpy
    x_true = np.linalg.solve(A, b)

    # Medir tiempo para Gauss sin pivoteo
    time_no_pivot, sol_no_pivot = measure_time(gauss_elimination, A, b)

    # Medir tiempo para Gauss con pivoteo parcial
    time_pivot, sol_pivot = measure_time(gauss_elimination_partial_pivoting, A, b)

    # Calcular errores
    error_no_pivot_abs, error_no_pivot_rel = calculate_errors(x_true, sol_no_pivot)
    error_pivot_abs, error_pivot_rel = calculate_errors(x_true, sol_pivot)

    results_time[f"Completa {size}x{size}"] = {
        "Sin pivoteo": time_no_pivot,
        "Con pivoteo": time_pivot
    }
    results_precision[f"Completa {size}x{size}"] = {
        "Error absoluto (sin pivoteo)": error_no_pivot_abs,
        "Error relativo (sin pivoteo)": error_no_pivot_rel,
        "Error absoluto (con pivoteo)": error_pivot_abs,
        "Error relativo (con pivoteo)": error_pivot_rel
    }

# Imprimir resultados de tiempo
print("Resultados de Tiempo de Ejecución:")
for key, value in results_time.items():
    print(f"Resultados para {key}:")
    for sub_key, sub_value in value.items():
        if isinstance(sub_value, float):
            print(f"  {sub_key}: {sub_value:.6f} segundos")
        else:
            print(f"  {sub_key}:")
            for sub_sub_key, sub_sub_value in sub_value.items():
                print(f"    {sub_sub_key}: {sub_sub_value:.6f}")
    print()

# Imprimir resultados de precisión
print("Resultados de Precisión:")
for key, value in results_precision.items():
    print(f"Resultados para {key}:")
    for sub_key, sub_value in value.items():
        if isinstance(sub_value, float):
            print(f"  {sub_key}: {sub_value:.6e}")
        else:
            print(f"  {sub_key}:")
            for sub_sub_key, sub_sub_value in sub_value.items():
                print(f"    {sub_sub_key}: {sub_sub_value:.6e}")
    print()
