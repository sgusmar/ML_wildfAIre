

# utils/data_utils.py

# src/utils/data_utils.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def guardar_dataset(df, nombre, carpeta="src/data_sample", con_fecha=False):
    """
    Guarda un DataFrame o Series de pandas en formato CSV.

    Parámetros:
    - df: DataFrame o Series a guardar
    - nombre: nombre del archivo (sin extensión)
    - carpeta: ruta donde guardar el archivo
    - con_fecha: si True, añade la fecha al nombre del archivo
    """
    os.makedirs(carpeta, exist_ok=True)

    if con_fecha:
        fecha = datetime.now().strftime("%Y%m%d")
        nombre = f"{nombre}_{fecha}"

    ruta = os.path.join(carpeta, f"{nombre}.csv")
    df.to_csv(ruta, index=False)

    n_filas = df.shape[0]
    n_columnas = df.shape[1] if df.ndim > 1 else 1  # Series -> 1 columna implícita

    print(f"Dataset guardado en: {ruta} ({n_filas} filas, {n_columnas} columnas)")

    return ruta


def plot_distribuciones(df, columnas=None, n_cols=3):
    """
    Grafica el histograma (con KDE) de cada columna numérica de un DataFrame.

    Parámetros:
    - df: DataFrame de pandas
    - columnas: lista de columnas a graficar (por defecto, todas las de df)
    - n_cols: número de columnas del grid de subplots
    """
    if columnas is None:
        columnas = df.columns

    n_rows = int(np.ceil(len(columnas) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columnas):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_distribuciones_por_clase(X, y, target_name="target", colores=None, ncols=2):
    """
    Grafica la distribución de cada feature de X, separada por clase del target y,
    con una línea vertical marcando la media de cada clase.

    Parámetros:
    - X: DataFrame con las features
    - y: Series con la variable objetivo (misma longitud que X)
    - target_name: nombre a mostrar para la variable objetivo en título/leyenda
    - colores: diccionario {valor_clase: color_hex}. Por defecto usa dos colores predefinidos.
    - ncols: número de columnas del grid de subplots
    """
    if colores is None:
        colores = {0: "#58ACE0FF", 1: "#8D44AC"}

    df = pd.concat([X, y.rename(target_name)], axis=1)
    features = X.columns.tolist()

    n_features = len(features)
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        for clase, color in colores.items():
            datos = df.loc[df[target_name] == clase, feat]
            axes[i].hist(
                datos,
                bins=20,
                color=color,
                alpha=0.5,
                label=str(clase),
                edgecolor="white",
                linewidth=0.5
            )

            axes[i].axvline(
                datos.mean(),
                color=color,
                linewidth=2,
                linestyle="--"
            )

        axes[i].set_title(f"Distribución de '{feat}' por {target_name}", fontweight="bold", fontsize=10)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Frecuencia")
        axes[i].legend(title=target_name.capitalize())

    # Oculta ejes sobrantes si el número de features no llena la última fila
    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"¿Cómo varía cada feature según {target_name}?", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_correlacion(X, y=None, target_name="target", excluir=None, metodo="spearman", cmap="BuPu", figsize=(12, 10)):
    """
    Calcula y grafica la matriz de correlación a partir de X (y opcionalmente y),
    excluyendo columnas categóricas u otras que no se quieran incluir.

    Parámetros:
    - X: DataFrame con las features
    - y: Series con la variable objetivo (opcional). Si se pasa, se incluye en la correlación.
    - target_name: nombre a asignar a la columna de y en la matriz de correlación
    - excluir: nombre de columna (str) o lista de columnas a excluir del cálculo
    - metodo: método de correlación -> "pearson", "spearman" o "kendall"
    - cmap: paleta de color del heatmap
    - figsize: tamaño de la figura
    """
    metodos_validos = ["pearson", "spearman", "kendall"]
    if metodo not in metodos_validos:
        raise ValueError(f"metodo debe ser uno de {metodos_validos}, recibido: '{metodo}'")

    if excluir is None:
        excluir = []
    elif isinstance(excluir, str):
        excluir = [excluir]

    if y is not None:
        df = pd.concat([X, y.rename(target_name)], axis=1)
    else:
        df = X.copy()

    columnas_continuas = [c for c in df.columns if c not in excluir]

    corr = df[columnas_continuas].corr(method=metodo)

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0)
    plt.title(f"Correlación de {metodo.capitalize()}")
    plt.tight_layout()
    plt.show()


def plot_boxplots(df, columnas=None, ncols=3, color="#58ACE0FF", titulo="Distribución y outliers por variable"):
    """
    Grafica un boxplot por cada columna numérica de un DataFrame, para
    visualizar su distribución y detectar outliers.

    Parámetros:
    - df: DataFrame de pandas
    - columnas: lista de columnas a graficar (por defecto, todas las de df)
    - ncols: número de columnas del grid de subplots
    - color: color de los boxplots
    - titulo: título general de la figura
    """
    if columnas is None:
        columnas = df.columns.tolist()

    n = len(columnas)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columnas):
        sns.boxplot(y=df[col], ax=axes[i], color=color)
        axes[i].set_title(col, fontweight="bold")

    for j in range(len(columnas), len(axes)):
        axes[j].axis("off")

    plt.suptitle(titulo, fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()