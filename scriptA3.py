import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Steel_industry_data.csv")
cols = ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor", "NSM"]
Xr = df[cols].values.astype(float)
y = df["Usage_kWh"].values.astype(float)
n = len(y)

# Solución exacta (ecuaciones normales, como en B.5)
X = np.column_stack([np.ones(n), Xr])
beta_exacta = np.linalg.solve(X.T @ X, X.T @ y)

# Estandarizamos las variables explicativas (media 0, desvío 1)
mu, sd = Xr.mean(axis=0), Xr.std(axis=0)
Xz = np.column_stack([np.ones(n), (Xr - mu) / sd])

def f(b):                      # f(b) = (1/2n) * suma de errores al cuadrado
    return np.sum((y - Xz @ b) ** 2) / (2 * n)

def grad(b):                   # gradiente de f
    return Xz.T @ (Xz @ b - y) / n

def descenso(alpha, eps=1e-10, max_iter=20000):
    b = np.zeros(Xz.shape[1])
    hist = [f(b)]
    for k in range(max_iter):
        b = b - alpha * grad(b)                    # b(k+1) = b(k) - alpha * grad f(b(k))
        hist.append(f(b))
        if not np.isfinite(hist[-1]) or hist[-1] > 1e12:   # divergencia
            break
        if abs(hist[-1] - hist[-2]) < eps:                 # criterio de corte
            break
    return b, hist

def volver_a_escala_original(bz):
    pendientes = bz[1:] / sd
    b0 = bz[0] - np.sum(bz[1:] * mu / sd)
    return np.concatenate([[b0], pendientes])

tasas = {"demasiado chica": 0.001, "adecuada": 0.2, "divergente": 1.1}
plt.figure(figsize=(8, 5))
for nombre, a in tasas.items():
    b, hist = descenso(a)
    b_orig = volver_a_escala_original(b)
    print(f"alpha={a} ({nombre}): iteraciones={len(hist)-1}, f final={hist[-1]:.4g}")
    print("   coeficientes:", b_orig)
    print("   diferencia máxima con la exacta:", np.max(np.abs(b_orig - beta_exacta)))
    plt.plot(hist, label=f"alpha = {a} ({nombre})")

plt.yscale("log"); plt.xscale("symlog")
plt.xlabel("Iteración k"); plt.ylabel("f(b)  (escala log)")
plt.legend(); plt.grid(alpha=0.3)
plt.xlim(left=0)
plt.savefig("A3-v2.png", dpi=120)
plt.show()
print("exacta:", beta_exacta)