#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparación, limpieza y transformación de datos para las tareas de clasificación:
  - 'pCR': respuesta patológica completa  (train-data-pCR.csv)
  - 'OS' : supervivencia global           (survival-data-OS.csv)
  - 'DFS': supervivencia libre de enfermedad (survival-data-DFS.csv)

Flujo de uso:
    X, y, feature_names, continuous_idx = prepare_task('pCR')
    ...
    # dentro de cada fold (para no filtrar información del test):
    X_tr, X_ts = scale_fold(X[train_index], X[test_index], continuous_idx)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuración por tarea
# ---------------------------------------------------------------------------
TASKS = {
    'pCR': {'file': 'train-data-pCR.csv',       'target': 'pCR'},
    'OS':  {'file': 'survival-data-OS.csv',     'target': 'OS'},
    'DFS': {'file': 'survival-data-DFS.csv',    'target': 'DFS'},
}

# Variables clínicas previas al tratamiento (disponibles en las tres bases)
PRE_TREATMENT = ['Menopausal_Status', 'Age', 'cT', 'cN', 'Histology', 'G',
                 'ER', 'PR', 'Ki67', 'HER2']
# Variables posteriores a la cirugía (solo en las bases de supervivencia)
POST_TREATMENT = ['pCR', 'ypT', 'ypN']

# Variables numéricas continuas y rangos válidos
NUMERIC_RANGES = {'Age': (18, 100), 'ER': (0, 100), 'PR': (0, 100), 'Ki67': (0, 100)}
CONTINUOUS = list(NUMERIC_RANGES)

# Categorías válidas (se fijan para que las columnas sean siempre las mismas)
ORDINAL_MAPS = {
    'cT':  {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4},
    'cN':  {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
    'ypT': {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4},
    'ypN': {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3},
    'G':   {'G1_2': 0, 'G3': 1},
}
BINARY_MAPS = {
    'Menopausal_Status': {'Premenopausal': 0, 'Postmenopausal': 1},
    'HER2': {'Zero': 0, 'Low': 1},
    'pCR':  {'No': 0, 'Yes': 1},
}
ONEHOT_CATEGORIES = {'Histology': ['Ductal', 'Lobular', 'Other']}


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def load_dataset(task, data_dir='.'):
    """Lee el CSV de la tarea ('pCR', 'OS' o 'DFS')."""
    if task not in TASKS:
        raise ValueError(f"Tarea desconocida '{task}'. Usa una de {list(TASKS)}.")
    path = f"{data_dir}/{TASKS[task]['file']}"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
def clean_data(df, task, verbose=True):
    """
    Limpieza determinista (no aprende nada de los datos, por lo que es seguro
    aplicarla antes de la validación cruzada):
      - normaliza texto (espacios) y nombres de columnas
      - elimina filas con etiqueta ausente
      - elimina duplicados exactos
      - valida categorías y rangos numéricos (valores inválidos -> NaN)
      - elimina filas con valores faltantes restantes
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    target = TASKS[task]['target']
    n0 = len(df)

    # Texto: quitar espacios
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype('string').str.strip()

    # Etiqueta ausente (hay 8 en OS y 16 en DFS)
    n_lbl = df[target].isna().sum()
    df = df.dropna(subset=[target])

    # Duplicados exactos
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()

    # Categorías fuera de catálogo -> NaN
    for col, mapping in {**ORDINAL_MAPS, **BINARY_MAPS}.items():
        if col in df.columns:
            df.loc[~df[col].isin(mapping.keys()), col] = pd.NA
    df.loc[~df['Histology'].isin(ONEHOT_CATEGORIES['Histology']), 'Histology'] = pd.NA

    # Numéricas fuera de rango -> NaN
    for col, (lo, hi) in NUMERIC_RANGES.items():
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # Faltantes restantes en las columnas usadas
    used = [c for c in PRE_TREATMENT + POST_TREATMENT + [target] if c in df.columns]
    n_na = df[used].isna().any(axis=1).sum()
    df = df.dropna(subset=used).reset_index(drop=True)

    if verbose:
        print(f"[clean_data:{task}] filas iniciales={n0} | sin etiqueta={n_lbl} | "
              f"duplicadas={n_dup} | con faltantes/invalidas={n_na} | finales={len(df)}")
    return df


# ---------------------------------------------------------------------------
# 3. Transformación (codificación)
# ---------------------------------------------------------------------------
def encode_features(df, include_post_treatment=True):
    """
    Codifica las variables a numéricas:
      - ordinales (cT, cN, ypT, ypN, G) -> enteros que respetan el orden
      - binarias (Menopausal_Status, HER2, pCR) -> 0/1
      - Histology -> one-hot con categorías fijas (Ductal, Lobular, Other)
      - numéricas continuas -> sin cambios (el escalado se hace por fold)
    Devuelve un DataFrame solo con las características.
    """
    cols = list(PRE_TREATMENT)
    if include_post_treatment:
        cols += [c for c in POST_TREATMENT if c in df.columns]

    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in ORDINAL_MAPS:
            out[col] = df[col].map(ORDINAL_MAPS[col]).astype(float)
        elif col in BINARY_MAPS:
            out[col] = df[col].map(BINARY_MAPS[col]).astype(float)
        elif col in ONEHOT_CATEGORIES:
            for cat in ONEHOT_CATEGORIES[col]:
                out[f'{col}_{cat}'] = (df[col] == cat).astype(float)
        else:  # numéricas continuas
            out[col] = df[col].astype(float)
    return out


def encode_target(series, task):
    """Convierte la etiqueta a vector 1D de enteros 0/1."""
    if task == 'pCR':
        return series.map(BINARY_MAPS['pCR']).astype(int).to_numpy()
    return series.astype(int).to_numpy()  # OS / DFS ya vienen como 0.0 / 1.0


# ---------------------------------------------------------------------------
# 4. Pipeline completo por tarea
# ---------------------------------------------------------------------------
def prepare_task(task, data_dir='.', include_post_treatment=True, verbose=True):
    """
    Carga, limpia y codifica la base de la tarea.

    Parámetros
    ----------
    task : 'pCR' | 'OS' | 'DFS'
    include_post_treatment : en OS/DFS incluye pCR, ypT y ypN como predictores
        (son resultados de la cirugía). En pCR se ignora, porque pCR es la
        etiqueta y no existen ypT/ypN en esa base.

    Devuelve
    --------
    X : ndarray (n, p)   características codificadas (sin escalar)
    y : ndarray (n,)     etiqueta binaria 0/1
    feature_names : list[str]
    continuous_idx : list[int]  posiciones de las columnas a escalar por fold
    """
    df = load_dataset(task, data_dir)
    df = clean_data(df, task, verbose=verbose)

    feats = encode_features(df, include_post_treatment=(include_post_treatment and task != 'pCR'))
    y = encode_target(df[TASKS[task]['target']], task)

    feature_names = list(feats.columns)
    continuous_idx = [feature_names.index(c) for c in CONTINUOUS if c in feature_names]

    if verbose:
        pos = y.mean() * 100
        print(f"[prepare_task:{task}] X={feats.shape} | y positivos={y.sum()} ({pos:.1f}%) | "
              f"negativos={(y == 0).sum()}")
    return feats.to_numpy(dtype=float), y, feature_names, continuous_idx


# ---------------------------------------------------------------------------
# 5. Escalado dentro de cada fold (evita fuga de información)
# ---------------------------------------------------------------------------
def scale_fold(X_train, X_test, continuous_idx):
    """
    Estandariza solo las columnas continuas, ajustando el escalador ÚNICAMENTE
    con el conjunto de entrenamiento del fold. Debe llamarse antes del
    sobremuestreo (SMOTE) y de entrenar el clasificador.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler().fit(X_train[:, continuous_idx])
    X_train[:, continuous_idx] = scaler.transform(X_train[:, continuous_idx])
    X_test[:, continuous_idx] = scaler.transform(X_test[:, continuous_idx])
    return X_train, X_test
