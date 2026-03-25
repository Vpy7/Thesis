import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
import seaborn as sns
from scipy.stats import chi2_contingency
from itertools import combinations
from pathlib import Path

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from reportlab.platypus import Image

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Advertencia: scipy no está instalado. No se generará el dendrograma.")


def cond_df(df, code, disp=False):
    code_cols = [i for i in df if i.startswith(code) and i[3].isnumeric()]
    root_cols = [True if len(i) <= 6 else False for i in code_cols]

    new_df = df[code_cols]
    df_root = pd.DataFrame({"question": code_cols, "root": root_cols})

    if disp:
        nan_display(new_df)

    return new_df, df_root

def nan_display(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    display(df.isnull().sum())

def comp_cols(df):
    rows = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        for val, cnt in counts.items():
            rows.append([col, val, cnt, cnt/len(df)*100])
    return pd.DataFrame(rows, columns=["question", "value", "count", "pct"])

def comp_cols_pct(df):
    possible_values = sorted(df.stack().dropna().unique())
    include_nan = df.isna().any().any()
    columns = ["question"]
    if include_nan:
        columns.append("pct_nan")
    columns += [f"pct_{val}" for val in possible_values]

    rows = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        row = [col]
        if include_nan:
            row.append(counts.get(np.nan, 0)/len(df)*100)
        row += [counts.get(val, 0)/len(df)*100 for val in possible_values]
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)

def comp_cols_pct_limited(df):
    allowed = {np.nan, 0, 2, 3, 7, 77, 8, 88, 9, 99}

    possible_values = sorted(v for v in df.stack().dropna().unique() if v in allowed)
    include_nan = df.isna().any().any()

    columns = ["question"]
    if include_nan:
        columns.append("pct_nan")
    columns += [f"pct_{val}" for val in possible_values]
    columns.append("limited")

    rows = []
    for col in df.columns:
        values = set(df[col].dropna().unique())
        is_limited = values.issubset(allowed - {np.nan})

        counts = df[col].value_counts(dropna=False)
        row = [col]

        if include_nan:
            row.append(counts.get(np.nan, 0) / len(df) * 100)

        row += [counts.get(val, 0) / len(df) * 100 for val in possible_values]
        row.append(is_limited)

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)

def plot_comp(df_comp, condition):
    map_labels = {
        2: "2 - Sí",
        0: "0 - No",
        3: "3 - A veces / Algo",
        7: "7 - Se niega a responder",
        77: "77 - Se niega a responder",
        8: "8 - No aplicable",
        88: "88 - No aplicable",
        9: "9 - No sabe",
        99: "99 - No sabe"
    }
    
    map_colors = {
        "2 - Sí": "#2ca02c",
        "0 - No": "#d62728",
        "3 - A veces / Algo": "#1f77b4",
        "7 - Se niega a responder": "#9467bd",
        "77 - Se niega a responder": "#9467bd",
        "8 - No aplicable": "#8c564b",
        "88 - No aplicable": "#8c564b",
        "9 - No sabe": "#e377c2",
        "99 - No sabe": "#e377c2",
        "NaN": "#7f7f7f"
    }


    questions = df_comp["question"].unique()
    fig = go.Figure()

    for q in questions:
        sub = df_comp[df_comp["question"] == q].copy()
        sub["label"] = sub["value"].apply(
            lambda v: map_labels[v] if v in map_labels else ("NaN" if pd.isna(v) else str(v))
        )
        sub["color"] = sub["label"].apply(lambda l: map_colors.get(l, "#7f7f7f"))

        fig.add_trace(
            go.Pie(
                labels=sub["label"],
                values=sub["count"],
                hole=0.5,
                visible=False,
                name=q,
                marker=dict(colors=sub["color"])
            )
        )

    fig.data[0].visible = True

    buttons = []
    for i, q in enumerate(questions):
        vis = [False]*len(questions)
        vis[i] = True
        buttons.append(
            dict(
                label=q,
                method="update",
                args=[{"visible": vis}]
            )
        )

    fig.update_layout(
        updatemenus=[{"buttons": buttons}],
        showlegend=True,
        title='Composición de Preguntas - ' + condition
    )

    return fig

def report(dic):
    pd.set_option('display.max_rows', None)
    for code, info in dic.items():

        print(info["nombre"], "-", code)
        print('')
        print('COMPOSICIÓN DE RESPUESTAS')
        
        display(info['df_comp'])
        display(info['comp_plot'])
        print('')
        print('NÚMERO DE RESPUESTAS BASALES Y NO BASALES')
        
        print('Preguntas Basales: ', info['root_question_count'])
        print('Preguntas Totales: ', info['question_count'])
        print('Porcentaje de Preguntas Basales: ', info['root_question_percent'], '%')
        print('')
        print('NÚMERO DE RESPUESTAS PERDIDAS')
        
        display(info['df_comp'][info['df_comp']['value'].isna()])

        print('-'*500)

def build_full_dict(df_sintomas_disc):
    full_dict = {
        "pag": {"nombre": "Agorafobia", "modulo": "A"},
        "psp": {"nombre": "Fobia específica", "modulo": "A"},
        "pso": {"nombre": "Fobia social", "modulo": "A"},
        "pga": {"nombre": "Ansiedad generalizada", "modulo": "A"},
        "psm": {"nombre": "Mutismo selectivo", "modulo": "A"},
        "ppa": {"nombre": "Pánico", "modulo": "A"},
        "ppt": {"nombre": "Trastorno por estrés postraumático", "modulo": "A"},
        "psa": {"nombre": "Ansiedad por separación", "modulo": "A"},
        "poc": {"nombre": "Trastorno obsesivo-compulsivo", "modulo": "A"},
        "pea": {"nombre": "Bulimia", "modulo": "B"},
        "pel": {"nombre": "Trastorno de eliminación", "modulo": "B"},
        "ppi": {"nombre": "Pica", "modulo": "B"},
        "ptc": {"nombre": "Trastorno de tics", "modulo": "B"},
        "ptr": {"nombre": "Tricotilomanía", "modulo": "B"},
        "pmd": {"nombre": "Depresión mayor o distimia", "modulo": "C"},
        "pma": {"nombre": "Manía o hipomanía", "modulo": "C"},
        "psz": {"nombre": "Esquizofrenia", "modulo": "D"},
        "pad": {"nombre": "Trastorno por déficit de atención e hiperactividad", "modulo": "E"},
        "pcd": {"nombre": "Trastorno de conducta", "modulo": "E"},
        "pod": {"nombre": "Trastorno de oposición desafiante", "modulo": "E"},
        "pal": {"nombre": "Abuso de alcohol", "modulo": "F"},
        "pmj": {"nombre": "Consumo de marihuana", "modulo": "F"},
        "psu": {"nombre": "Consumo de otras sustancias", "modulo": "F"},
        "pni": {"nombre": "Consumo de tabaco", "modulo": "F"}
    }

    for code, info in full_dict.items():
        df_filtered, df_root = cond_df(df_sintomas_disc, code, disp=False)
        info["df"] = df_filtered
        df_comp = comp_cols(df_filtered)
        info["df_comp"] = df_comp
        info["df_comp_pct"] = comp_cols_pct(df_filtered)
        info["df_comp_pct_limited"] = comp_cols_pct_limited(df_filtered)
        info['df_comp_full'] = comp_cols_full(df_filtered)
        info['df_comp_full_limited'] = comp_cols_full_limited(df_filtered)
        info["question_count"] = len(df_root)



    return full_dict

def _fmt_val(v):
    return str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)


def comp_cols_full(df):
    import numpy as np
    import pandas as pd

    standard_map = {
        2: "SI", 0: "NO", 3: "AV/A",
        7: "NR", 77: "NR",
        8: "NA", 88: "NA",
        9: "NS", 99: "NS"
    }

    standard_vals = set(standard_map.keys())

    possible_values = sorted(df.stack().dropna().unique())
    include_nan = df.isna().any().any()

    columns = [
        "question",
        "cnt_non_nan", "pct_non_nan",
        "cnt_non_standard", "pct_non_standard"
    ]

    if include_nan:
        columns += ["cnt_nan", "pct_nan"]

    for v in possible_values:
        v_fmt = _fmt_val(v)
        suffix = f"{v_fmt}_{standard_map[v]}" if v in standard_map else v_fmt
        columns += [f"cnt_{suffix}", f"pct_{suffix}"]

    rows = []

    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        total = len(df)

        cnt_non_nan = counts.drop(index=[np.nan], errors="ignore").sum()
        cnt_non_standard = sum(
            c for v, c in counts.items()
            if not pd.isna(v) and v not in standard_vals
        )

        row = [
            col,
            cnt_non_nan, cnt_non_nan / total * 100,
            cnt_non_standard, cnt_non_standard / total * 100
        ]

        if include_nan:
            nan_cnt = counts.get(np.nan, 0)
            row += [nan_cnt, nan_cnt / total * 100]

        for v in possible_values:
            cnt = counts.get(v, 0)
            row += [cnt, cnt / total * 100]

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def comp_cols_full_limited(df):
    import numpy as np
    import pandas as pd

    standard_map = {
        2: "SI", 0: "NO", 3: "AV/A",
        7: "NR", 77: "NR",
        8: "NA", 88: "NA",
        9: "NS", 99: "NS"
    }

    standard_vals = set(standard_map.keys())

    possible_values = sorted(
        v for v in df.stack().dropna().unique() if v in standard_vals
    )
    include_nan = df.isna().any().any()

    columns = [
        "question",
        "cnt_non_nan", "pct_non_nan",
        "cnt_non_standard", "pct_non_standard"
    ]

    if include_nan:
        columns += ["cnt_nan", "pct_nan"]

    for v in possible_values:
        v_fmt = _fmt_val(v)
        suffix = f"{v_fmt}_{standard_map[v]}"
        columns += [f"cnt_{suffix}", f"pct_{suffix}"]

    columns.append("limited")

    rows = []

    for col in df.columns:
        counts = df[col].value_counts(dropna=False)
        total = len(df)

        cnt_non_nan = counts.drop(index=[np.nan], errors="ignore").sum()
        cnt_non_standard = sum(
            c for v, c in counts.items()
            if not pd.isna(v) and v not in standard_vals
        )

        values = set(df[col].dropna().unique())
        is_limited = values.issubset(standard_vals)

        row = [
            col,
            cnt_non_nan, cnt_non_nan / total * 100,
            cnt_non_standard, cnt_non_standard / total * 100
        ]

        if include_nan:
            nan_cnt = counts.get(np.nan, 0)
            row += [nan_cnt, nan_cnt / total * 100]

        for v in possible_values:
            cnt = counts.get(v, 0)
            row += [cnt, cnt / total * 100]

        row.append(is_limited)
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)

def tol_report(full_dict, max_tolerancias=2):
    for code, info in full_dict.items():
        df = info['df_comp_pct_limited']
        nombre = info['nombre']
        nans = sorted(df['pct_nan'].tolist())
        subset_total = pd.DataFrame()
        mostradas = 0

        print(f"{nombre} - {code} - N={len(nans)}")

        for i in range(5, 30):
            if mostradas >= max_tolerancias:
                break

            upper_idx = int(len(nans) * i * 0.01)
            lower_idx = int(len(nans) * (i - 5) * 0.01)

            upper_threshold = nans[min(upper_idx, len(nans) - 1)]
            lower_threshold = nans[max(lower_idx, 0)]

            new_subset = df[
                (df['pct_nan'] >= lower_threshold) &
                (df['pct_nan'] < upper_threshold)
            ]
            new_rows = new_subset[~new_subset.index.isin(subset_total.index)]

            if not new_rows.empty:
                print(f"Tolerancia de {i}% => Umbral de {lower_threshold:.2f} a {upper_threshold:.2f}")
                subset_total = pd.concat([subset_total, new_rows])
                display(new_rows)
                mostradas += 1

        print('\n')
        print('-' * 150)
        print('\n')
        
def imprimir_etiquetas(variables, df_etiquetas, col_codigo='Código', col_etiqueta='Etiqueta'):
    map_codigo_etiqueta = {cod.upper(): etiqueta 
                          for cod, etiqueta in zip(df_etiquetas[col_codigo], 
                                                  df_etiquetas[col_etiqueta])}
    
    print("\n" + "=" * 120)
    print(f"LISTADO DE ETIQUETAS ({len(variables)} variables)")
    print("=" * 120)
    
    encontradas = 0
    no_encontradas = []
    
    for i, var in enumerate(variables, 1):
        var_upper = var.upper()
        
        if var_upper in map_codigo_etiqueta:
            etiqueta = map_codigo_etiqueta[var_upper]
            encontradas += 1
            print(f"{i:3d}. [{var}] - {etiqueta}")
        else:
            print(f"{i:3d}. [{var}] - Variable no encontrada")
            no_encontradas.append(var)
    
    print("\n" + "=" * 120)
    print(f"RESUMEN:")
    print(f"  • Total variables: {len(variables)}")
    print(f"  • Encontradas: {encontradas}")
    print(f"  • No encontradas: {len(no_encontradas)}")
    
    if no_encontradas:
        print(f"\nVariables no encontradas: {', '.join(no_encontradas)}")
    
    print("=" * 120)

CODE_MAP = {
    0: "NO", 2: "SI", 3: "AV", 1: "AV",
    7: "NR", 77: "NR",
    8: "NA", 88: "NA", 
    9: "NS", 99: "NS"
}

def calculate_imc_categories(df, weight_col='pea002k', min_weight_col='pea003k', height_col='pea001z'):
    df = df.copy()
    
    required_cols = [weight_col, min_weight_col, height_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Advertencia: Columnas no encontradas: {missing_cols}")
        return df
    
    df['height_m'] = df[height_col] / 100
    
    df['IMC_pea002'] = df[weight_col] / (df['height_m'] ** 2)
    df['IMC_pea003'] = df[min_weight_col] / (df['height_m'] ** 2)
    
    imc_categories = ['underweight', 'normal weight', 'overweight', 'obese']
    imc_bins = [0, 18.5, 25, 30, np.inf]
    
    df['IMC_pea002'] = pd.cut(
        df['IMC_pea002'], 
        bins=imc_bins, 
        labels=imc_categories, 
        right=False
    )
    
    df['IMC_pea003'] = pd.cut(
        df['IMC_pea003'], 
        bins=imc_bins, 
        labels=imc_categories, 
        right=False
    )
    
    df = df.drop(columns=['height_m', weight_col, min_weight_col, height_col])
    
    return df


def apply_code_map(df, columns, code_map=CODE_MAP):
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: code_map.get(x, x) if pd.notnull(x) else x)
    
    return df


def perform_lca_clustering(df, question_list, id_column='id', n_components_range=range(2, 7)):
    warnings.filterwarnings('ignore')
    
    print("="*70)
    print("ANÁLISIS DE CLASES LATENTES (LCA)")
    print("="*70)
    
    work_df = df.copy()
    print(f"\n[1] Preparando datos: {len(work_df)} observaciones")
    
    imc_required_cols = ['pea001z', 'pea002k', 'pea003k']
    imc_cols_in_question_list = [col for col in imc_required_cols if col in question_list]
    
    if imc_cols_in_question_list and all(col in work_df.columns for col in imc_required_cols):
        print(f"[2] Procesando columnas de IMC...")
        print(f"    Transformando peso y estatura a categorías de IMC...")
        
        work_df = calculate_imc_categories(work_df)
        
        for col in imc_required_cols:
            if col in question_list:
                question_list.remove(col)
        
        imc_new_cols = ['IMC_pea002', 'IMC_pea003']
        for new_col in imc_new_cols:
            if new_col in work_df.columns:
                question_list.append(new_col)
        
        print(f"    ✓ IMC categorizado: {imc_new_cols}")
    
    print(f"\n[3] Aplicando mapeo de códigos...")
    
    existing_cols = [col for col in question_list if col in work_df.columns]
    missing_cols = [col for col in question_list if col not in work_df.columns]
    
    if missing_cols:
        print(f"    ⚠ Columnas no encontradas: {missing_cols}")
        question_list = existing_cols
    
    work_df = apply_code_map_without_imputation(work_df, question_list)
    print(f"    ✓ Mapeo aplicado a {len(question_list)} columnas")
    
    print(f"\n[4] Verificando tipos de datos...")
    non_categorical = []
    for col in question_list:
        if col in work_df.columns:
            dtype = work_df[col].dtype
            if dtype not in ['object', 'category']:
                non_categorical.append((col, dtype))
    
    if non_categorical:
        print(f"    ⚠ ADVERTENCIA: Columnas no categóricas encontradas:")
        for col, dtype in non_categorical:
            print(f"       {col}: {dtype}")
        print(f"    Estas columnas se forzarán a categóricas para LCA")
    
    print(f"\n[5] Análisis de valores faltantes (NaNs):")
    nan_df = work_df[question_list].copy()
    
    for col in nan_df.columns:
        nan_count = nan_df[col].isna().sum()
        if nan_count > 0:
            nan_percent = (nan_count / len(nan_df)) * 100
            print(f"    {col:<15}: {nan_count:>4} NaNs ({nan_percent:>6.2f}%)")
    
    total_nans = nan_df.isna().sum().sum()
    total_cells = nan_df.shape[0] * nan_df.shape[1]
    total_nan_percent = (total_nans / total_cells) * 100
    print(f"\n    Total: {total_nans:,} NaNs en {total_cells:,} celdas ({total_nan_percent:.2f}%)")
    
    print(f"\n[6] Preparando datos para LCA (todas variables como categóricas)...")
    
    cluster_df = work_df[question_list].copy()
    
    for col in cluster_df.columns:
        cluster_df[col] = cluster_df[col].astype(str)
    
    le_dict = {}
    encoded_data = []
    feature_names = []
    
    for col in cluster_df.columns:
        le = LabelEncoder()
        col_data = cluster_df[col].copy()
        col_data = col_data.fillna('MISSING')
        
        try:
            encoded = le.fit_transform(col_data)
            encoded_data.append(encoded)
            feature_names.append(col)
            le_dict[col] = le
        except Exception as e:
            print(f"    ✗ Error codificando {col}: {e}")
    
    if not encoded_data:
        print("    ✗ No se pudieron codificar datos para LCA")
        return work_df, None
    
    X = np.column_stack(encoded_data)
    print(f"    ✓ Datos preparados: {X.shape[0]} × {X.shape[1]}")
    print(f"    ✓ Columnas codificadas: {len(feature_names)}")
    
    print(f"\n[7] Realizando Análisis de Clases Latentes...")
    
    results = {}
    metrics = []
    prob_results = {}
    
    for n_components in n_components_range:
        print(f"\n    k = {n_components} clases latentes...")
        
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                random_state=42,
                n_init=10,
                max_iter=500,
                tol=1e-3
            )
            
            gmm.fit(X)
            clusters = gmm.predict(X)
            probabilities = gmm.predict_proba(X)
            
            results[n_components] = clusters
            prob_results[n_components] = probabilities
            
            max_probs = probabilities.max(axis=1)
            avg_certainty = max_probs.mean()
            
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
            metrics.append({
                'k': n_components,
                'BIC': bic,
                'AIC': aic,
                'Convergió': gmm.converged_,
                'Iteraciones': gmm.n_iter_,
                'Certeza_prom': avg_certainty
            })
            
            print(f"      ✓ BIC: {bic:.2f}, AIC: {aic:.2f}")
            print(f"      ✓ Certeza promedio: {avg_certainty:.3f}")
            
            unique, counts = np.unique(clusters, return_counts=True)
            print(f"      ✓ Distribución:")
            for cluster, count in sorted(zip(unique, counts)):
                percentage = (count / len(clusters)) * 100
                print(f"          Clase {cluster}: {count} ({percentage:.1f}%)")
            
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            results[n_components] = None
            prob_results[n_components] = None
    
    print(f"\n[8] Creando DataFrames de salida...")
    
    output_cols = [id_column] + question_list
    available_cols = [col for col in output_cols if col in work_df.columns]
    output_df = work_df[available_cols].copy()
    
    prob_df = pd.DataFrame()
    prob_df[id_column] = work_df[id_column].copy()
    
    for n_components in n_components_range:
        if results.get(n_components) is not None:
            col_name = f'lca_k{n_components}'
            output_df[col_name] = results[n_components]
            print(f"     Añadida columna: {col_name}")
            
            probabilities = prob_results[n_components]
            for class_idx in range(n_components):
                prob_col = f'lca_k{n_components}_prob_class{class_idx}'
                prob_df[prob_col] = probabilities[:, class_idx]
            print(f"     Añadidas {n_components} columnas de probabilidad para k={n_components}")
    
    print(f"\n" + "="*70)
    print("RESUMEN FINAL - LCA CON VARIABLES CATEGÓRICAS")
    print("="*70)
    
    print(f"\nDataFrame principal (output_df):")
    print(f"  • Dimensiones: {output_df.shape[0]} filas × {output_df.shape[1]} columnas")
    print(f"  • Variables categóricas: {len(question_list)}")
    
    print(f"\nDataFrame de probabilidades (prob_df):")
    print(f"  • Dimensiones: {prob_df.shape[0]} filas × {prob_df.shape[1]} columnas")
    print(f"  • Columnas de probabilidad: {prob_df.shape[1] - 1}")
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        print(f"\nMétricas de modelos:")
        print(metrics_df.to_string(index=False))
        
        valid_metrics = metrics_df[metrics_df['BIC'].notnull()]
        if not valid_metrics.empty:
            best_model = valid_metrics.loc[valid_metrics['BIC'].idxmin()]
            print(f"\n  → Modelo recomendado: k = {best_model['k']} (menor BIC, certeza={best_model['Certeza_prom']:.3f})")
    
    print(f"\n" + "="*70)
    print("PROCESO COMPLETADO")
    print("="*70)
    
    return output_df, prob_df

def question_list():
    codes = {
        "pso": [(1, 5)],
        "pga": [(1, 5), (23, 29)],
        "psa": [(1, 12)],
        "pmd": [(1, 22), (35, 35)],
        "psz": [(1, 2), (9, 14)],
        "pad": [(1, 2), (4, 11), (22, 25), (27, 29), (31, 33), (44, 44)],
        "pcd": [(1, 4), (7, 9), (12, 15), (19, 23), (25, 25), (27, 29), (39, 39)],
        "pod": [(1, 12)],
        "pal": [(1, 1)],
        "pmj": [(1, 1)],
        "psu": [(1, 11)],
        "pni": [(1, 2)]
    }
    
    replacements = {
        'pea001': 'pea001z',
        'pea002': 'pea002k',
        'pea003': 'pea003k'
    }
    
    return [
        replacements.get(f"{code}{num:03d}", f"{code}{num:03d}")
        for code, ranges in codes.items()
        for start, end in ranges
        for num in range(start, end + 1)
    ]


def apply_code_map_without_imputation(df, columns, code_map=CODE_MAP):
    df_result = df.copy()
    
    for col in columns:
        if col in df_result.columns:
            df_result[col] = df_result[col].apply(
                lambda x: code_map.get(x, x) if pd.notnull(x) else x
            )
    
    return df_result

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def analisis_estadistico_clusters(final_df, id_column='id', cluster_prefix='lca_k',
                                   umbral_diferencia=0.10, top_n_variables=25, 
                                   umbral_asociacion=0.15, df_etiquetas=None,
                                   col_codigo='Código', col_etiqueta='Etiqueta',
                                   output_pdf=None, metrics_df=None,
                                   mostrar_todas_variables_perfil=True,
                                   max_variables_perfil=None):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    np.set_printoptions(precision=3, suppress=True)
    
    print("=" * 100)
    print("ANÁLISIS ESTADÍSTICO DE CLASES")
    print("=" * 100)
    
    cluster_cols = sorted([col for col in final_df.columns if cluster_prefix in col])
    print(f"\nCOLUMNAS DE CLASE ENCONTRADAS: {cluster_cols}")
    print(f"Total de observaciones: {len(final_df)}")
    
    cat_vars = [col for col in final_df.columns 
                if col not in cluster_cols and col != id_column]
    
    print(f"\nTOTAL variables para análisis: {len(cat_vars)}")
    
    mapeo_etiquetas = {}
    if df_etiquetas is not None:
        for _, row in df_etiquetas.iterrows():
            codigo = str(row[col_codigo]).strip()
            etiqueta = str(row[col_etiqueta]).strip()
            mapeo_etiquetas[codigo.upper()] = etiqueta
        print(f"  ✓ Se cargaron {len(mapeo_etiquetas)} etiquetas")
    
    def obtener_etiqueta(var_codigo):
        clave = var_codigo.upper()
        return mapeo_etiquetas.get(clave, None)
    
    resultados = {}
    if metrics_df is not None:
        resultados['metrics'] = metrics_df
    
    print("\n" + "=" * 100)
    print("INICIANDO ANÁLISIS COMPLETO...")
    print("=" * 100)
    
    resultados['distribucion'] = analizar_distribucion_clusters(final_df, cluster_cols, cat_vars)
    
    variables_clave, cramers_completos = identificar_variables_clave(
        final_df, cluster_cols, cat_vars, top_n_variables, obtener_etiqueta)
    resultados['variables_clave'] = variables_clave
    resultados['cramers_completos'] = cramers_completos
    
    resultados['perfiles'] = analizar_perfiles_clusters(
        final_df, cluster_cols, cat_vars, top_n_variables, umbral_diferencia, 
        variables_clave, obtener_etiqueta,
        mostrar_todas_variables=mostrar_todas_variables_perfil,
        max_variables_perfil=max_variables_perfil)
    
    resultados['jerarquia'] = analizar_jerarquia_clusters(final_df, cluster_cols)
    
    print("\n" + "=" * 100)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 100)
    
    if output_pdf:
        try:
            generar_reporte_pdf(resultados, output_pdf, final_df, cat_vars, cluster_cols, obtener_etiqueta)
        except Exception as e:
            print(f"Error al generar PDF: {e}")
    
    return resultados


def analizar_distribucion_clusters(final_df, cluster_cols, cat_vars):
    print("\n" + "=" * 80)
    print("1. ESTADÍSTICAS DE DISTRIBUCIÓN DE CLASES POR AGRUPACIÓN")
    print("=" * 80)
    
    resumen_clases = pd.DataFrame()
    
    for cluster_col in cluster_cols:
        cluster_stats = final_df[cluster_col].value_counts().sort_index()
        
        for cluster_id in cluster_stats.index:
            cluster_data = final_df[final_df[cluster_col] == cluster_id]
            cluster_size = len(cluster_data)
            
            missing_stats = []
            for var in cat_vars:
                if var in cluster_data.columns:
                    missing_pct = cluster_data[var].isna().mean() * 100
                    missing_stats.append(missing_pct)
            
            avg_missing = np.mean(missing_stats) if missing_stats else 100
            
            resumen_clases = pd.concat([
                resumen_clases,
                pd.DataFrame([{
                    'Agrupacion': cluster_col,
                    'Clase': cluster_id,
                    'N_Observaciones': cluster_size,
                    'Porcentaje_Total': (cluster_size / len(final_df)) * 100,
                    'Missing_Promedio': avg_missing,
                    'Completitud_Promedio': 100 - avg_missing
                }])
            ])
    
    print("\nRESUMEN DE TODAS LAS CLASES:")
    for cluster_col in cluster_cols:
        cluster_data = resumen_clases[resumen_clases['Agrupacion'] == cluster_col]
        total_obs = cluster_data['N_Observaciones'].sum()
        
        print(f"\n{cluster_col}:")
        print(f"  Número de clases: {len(cluster_data)}")
        print(f"  Total observaciones: {total_obs}")
        print(f"  Tamaño promedio de clase: {cluster_data['N_Observaciones'].mean():.0f}")
        print(f"  Rango de tamaños: {cluster_data['N_Observaciones'].min():.0f} - {cluster_data['N_Observaciones'].max():.0f}")
        print(f"  Completitud promedio: {cluster_data['Completitud_Promedio'].mean():.1f}%")
    
    return resumen_clases


def identificar_variables_clave(final_df, cluster_cols, cat_vars, top_n_variables, obtener_etiqueta):
    print("\n" + "=" * 80)
    print("2. IDENTIFICACIÓN SISTEMÁTICA DE VARIABLES CLAVE (TODAS LAS VARIABLES)")
    print("=" * 80)
    
    variables_discriminantes_global = {}
    cramers_per_k_dict = {}
    
    for cluster_col in cluster_cols:
        print(f"\nAnalizando {len(cat_vars)} variables para {cluster_col}...")
        
        discriminacion_scores = []
        
        for var in cat_vars:
            if var in final_df.columns:
                try:
                    var_data = final_df[var].copy()
                    
                    if pd.api.types.is_numeric_dtype(var_data):
                        if var_data.nunique() > 4:
                            var_data = pd.qcut(var_data, 4, duplicates='drop')
                        else:
                            var_data = var_data.astype(str)
                    else:
                        var_data = var_data.astype(str)
                    
                    contingency_table = pd.crosstab(var_data, final_df[cluster_col])
                    
                    if contingency_table.shape[0] <= 1 or contingency_table.shape[1] <= 1:
                        continue
                    
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    
                    discriminacion_total = 0
                    for cluster_id in contingency_table.columns:
                        cluster_vals = contingency_table[cluster_id]
                        total_cluster = cluster_vals.sum()
                        if total_cluster > 0:
                            top_2_cats = cluster_vals.nlargest(2)
                            concentracion = top_2_cats.sum() / total_cluster
                            discriminacion_total += concentracion
                    
                    discriminacion_promedio = discriminacion_total / len(contingency_table.columns) if len(contingency_table.columns) > 0 else 0
                    
                    discriminacion_scores.append({
                        'Variable': var,
                        'Cramers_V': cramers_v,
                        'p_value': p,
                        'Chi2': chi2,
                        'Discriminacion_Promedio': discriminacion_promedio,
                        'Significativo': p < 0.001
                    })
                    
                except Exception:
                    continue
        
        discriminacion_df = pd.DataFrame(discriminacion_scores)
        
        if len(discriminacion_df) == 0:
            print(f"  No se pudo analizar ninguna variable para {cluster_col}")
            cramers_per_k_dict[cluster_col] = pd.DataFrame(columns=['Variable', 'Cramér\'s V', 'p-value', 'χ²'])
            variables_discriminantes_global[cluster_col] = pd.DataFrame()
            continue
        
        cramers_completo = discriminacion_df[['Variable', 'Cramers_V', 'p_value', 'Chi2']].copy()
        cramers_completo.columns = ['Variable', 'Cramér\'s V', 'p-value', 'χ²']
        cramers_per_k_dict[cluster_col] = cramers_completo
        
        variables_significativas = discriminacion_df[discriminacion_df['Significativo']].copy()
        variables_significativas = variables_significativas.sort_values('Cramers_V', ascending=False)
        
        print(f"  Variables significativas encontradas: {len(variables_significativas)} de {len(cat_vars)}")
        
        if len(variables_significativas) > 0:
            variables_discriminantes_global[cluster_col] = variables_significativas
            
            print(f"\n  Top {top_n_variables} variables más discriminantes para {cluster_col}:")
            print("  " + "-" * 120)
            print("  {:<25} {:<40} {:>10} {:>12} {:>10} {:>12}".format(
                "Variable", "Etiqueta", "Cramér's V", "p-value", "χ²", "Discriminación"))
            print("  " + "-" * 120)
            
            for idx, row in variables_significativas.head(top_n_variables).iterrows():
                var_codigo = row['Variable']
                etiqueta = obtener_etiqueta(var_codigo) or ""
                cramer = row['Cramers_V']
                pval = row['p_value']
                chi2 = row['Chi2']
                disc = row['Discriminacion_Promedio']
                
                print("  {:<25} {:<40} {:>10.4f} {:>12.2e} {:>10.0f} {:>12.3f}".format(
                    var_codigo[:25], etiqueta[:40], cramer, pval, chi2, disc))
        else:
            print(f"  No se encontraron variables significativas (p < 0.001) para {cluster_col}")
            variables_discriminantes_global[cluster_col] = pd.DataFrame()
        
        umbrales = [0.3, 0.4, 0.5]
        for umbral in umbrales:
            print(f"\n  Variables con Cramér's V > {umbral}:")
            vars_umbral = discriminacion_df[discriminacion_df['Cramers_V'] > umbral].sort_values('Cramers_V', ascending=False)
            if len(vars_umbral) > 0:
                print("  " + "-" * 110)
                print("  {:<25} {:<40} {:>10} {:>12} {:>10}".format(
                    "Variable", "Etiqueta", "Cramér's V", "p-value", "χ²"))
                print("  " + "-" * 110)
                for _, row in vars_umbral.iterrows():
                    var_codigo = row['Variable']
                    etiqueta = obtener_etiqueta(var_codigo) or ""
                    cramer = row['Cramers_V']
                    pval = row['p_value']
                    chi2 = row['Chi2']
                    print("  {:<25} {:<40} {:>10.4f} {:>12.2e} {:>10.0f}".format(
                        var_codigo[:25], etiqueta[:40], cramer, pval, chi2))
            else:
                print("    No hay variables con Cramér's V > {}".format(umbral))
    
    return variables_discriminantes_global, cramers_per_k_dict


def analizar_perfiles_clusters(final_df, cluster_cols, cat_vars, top_n_variables, umbral_diferencia, 
                               variables_discriminantes_global, obtener_etiqueta, 
                               mostrar_todas_variables=True, max_variables_perfil=None):
    print("\n" + "=" * 80)
    print(f"3. PERFIL COMPLETO POR CLASE (ANÁLISIS CON {'TODAS' if mostrar_todas_variables else str(top_n_variables) + ' VARIABLES MÁS DISCRIMINANTES'})")
    print("=" * 80)
    
    resultados_perfiles = {}
    
    for cluster_col in cluster_cols:
        print(f"\n{'#' * 70}")
        print(f"PERFILES DETALLADOS PARA LA AGRUPACIÓN: {cluster_col}")
        print(f"{'#' * 70}")
        
        if mostrar_todas_variables:
            variables_a_analizar = cat_vars
            print(f"Analizando todas las {len(variables_a_analizar)} variables...")
        else:
            if cluster_col in variables_discriminantes_global:
                variables_a_analizar = variables_discriminantes_global[cluster_col].head(top_n_variables)['Variable'].tolist()
            else:
                variables_a_analizar = cat_vars[:top_n_variables]
            print(f"Analizando {len(variables_a_analizar)} variables más discriminantes...")
        
        total_dataset = len(final_df)
        
        perfiles_detallados = []
        
        for cluster_id in sorted(final_df[cluster_col].unique()):
            cluster_data = final_df[final_df[cluster_col] == cluster_id]
            cluster_size = len(cluster_data)
            
            perfil_clase = {
                'Clase': cluster_id,
                'Tamaño': cluster_size,
                'Porcentaje': (cluster_size / total_dataset) * 100,
                'Sobre': [],
                'Sub': []
            }
            
            for var in variables_a_analizar:
                if var not in cluster_data.columns or var not in final_df.columns:
                    continue
                
                var_data_cluster = cluster_data[var].copy()
                var_data_global = final_df[var].copy()
                
                if pd.api.types.is_numeric_dtype(var_data_cluster):
                    try:
                        var_data_cluster = pd.qcut(var_data_cluster, 4, duplicates='drop')
                        var_data_global = pd.qcut(var_data_global, 4, duplicates='drop')
                    except:
                        var_data_cluster = var_data_cluster.astype(str)
                        var_data_global = var_data_global.astype(str)
                else:
                    var_data_cluster = var_data_cluster.astype(str)
                    var_data_global = var_data_global.astype(str)
                
                cluster_counts = var_data_cluster.value_counts()
                cluster_dist = cluster_counts / cluster_size
                global_counts = var_data_global.value_counts()
                global_dist = global_counts / total_dataset
                
                cluster_dist_filtrado = cluster_dist[cluster_dist >= 0.05]
                if len(cluster_dist_filtrado) == 0:
                    cluster_dist_filtrado = cluster_dist.head(3)
                
                for cat, freq_cluster_rel in cluster_dist_filtrado.items():
                    freq_global_rel = global_dist.get(cat, 0)
                    count_cluster = cluster_counts.get(cat, 0)
                    count_global = global_counts.get(cat, 0)
                    diferencia = freq_cluster_rel - freq_global_rel
                    
                    if diferencia > umbral_diferencia:
                        perfil_clase['Sobre'].append({
                            'Variable': var,
                            'Etiqueta': obtener_etiqueta(var) or "",
                            'Categoria': str(cat),
                            'Freq_Cluster_Rel': freq_cluster_rel,
                            'Count_Cluster': count_cluster,
                            'Freq_Global_Rel': freq_global_rel,
                            'Count_Global': count_global,
                            'Diferencia': diferencia
                        })
                    elif diferencia < -umbral_diferencia:
                        perfil_clase['Sub'].append({
                            'Variable': var,
                            'Etiqueta': obtener_etiqueta(var) or "",
                            'Categoria': str(cat),
                            'Freq_Cluster_Rel': freq_cluster_rel,
                            'Count_Cluster': count_cluster,
                            'Freq_Global_Rel': freq_global_rel,
                            'Count_Global': count_global,
                            'Diferencia': diferencia
                        })
            
            perfil_clase['Sobre'].sort(key=lambda x: abs(x['Diferencia']), reverse=True)
            perfil_clase['Sub'].sort(key=lambda x: abs(x['Diferencia']), reverse=True)
            
            if max_variables_perfil is not None:
                perfil_clase['Sobre'] = perfil_clase['Sobre'][:max_variables_perfil * 3]
                perfil_clase['Sub'] = perfil_clase['Sub'][:max_variables_perfil * 3]
            
            perfiles_detallados.append(perfil_clase)
            
            print(f"\n  CLASE {perfil_clase['Clase']}: {perfil_clase['Tamaño']} obs ({perfil_clase['Porcentaje']:.1f}%)")
            print(f"  {'─' * 100}")
            
            if not perfil_clase['Sobre'] and not perfil_clase['Sub']:
                print(f"  No se encontraron variables con diferencias >{umbral_diferencia*100:.0f}% respecto al global.")
            else:
                if perfil_clase['Sobre']:
                    print(f"  SOBRERREPRESENTADAS: {len(perfil_clase['Sobre'])} categorías")
                    vars_mostradas = set()
                    for item in perfil_clase['Sobre'][:10]:
                        if item['Variable'] not in vars_mostradas:
                            print(f"    - {item['Variable']}: {item['Categoria'][:30]} (+{item['Diferencia']*100:.1f}pp)")
                            vars_mostradas.add(item['Variable'])
                if perfil_clase['Sub']:
                    print(f"  SUBREPRESENTADAS: {len(perfil_clase['Sub'])} categorías")
                    vars_mostradas = set()
                    for item in perfil_clase['Sub'][:10]:
                        if item['Variable'] not in vars_mostradas:
                            print(f"    - {item['Variable']}: {item['Categoria'][:30]} ({item['Diferencia']*100:.1f}pp)")
                            vars_mostradas.add(item['Variable'])
            
            print(f"  {'─' * 100}")
        
        resultados_perfiles[cluster_col] = perfiles_detallados
    
    return resultados_perfiles


def analizar_patrones_combinados(final_df, cluster_cols, cat_vars, top_n_variables, umbral_asociacion, variables_discriminantes_global, obtener_etiqueta):
    print("\n" + "=" * 80)
    print(f"4. ANÁLISIS DE PATRONES COMBINADOS E INTERACCIONES ENTRE {top_n_variables} VARIABLES CLAVE")
    print("=" * 80)
    
    resultados_patrones = {}
    
    for cluster_col in cluster_cols:
        print(f"\n{'#' * 70}")
        print(f"ANÁLISIS DE PATRONES COMBINADOS PARA: {cluster_col}")
        print(f"{'#' * 70}")
        
        if cluster_col in variables_discriminantes_global:
            top_vars = variables_discriminantes_global[cluster_col].head(top_n_variables)['Variable'].tolist()
            print(f"\n  Variables analizadas ({top_n_variables} más discriminantes):")
            for i in range(0, top_n_variables, 3):
                grupo = top_vars[i:min(i+3, top_n_variables)]
                grupo_con_etiquetas = []
                for var in grupo:
                    etiqueta = obtener_etiqueta(var)
                    if etiqueta:
                        grupo_con_etiquetas.append(f"{var} ({etiqueta[:30]})")
                    else:
                        grupo_con_etiquetas.append(var)
                print(f"    {i+1:2d}-{min(i+3, top_n_variables):2d}: {', '.join(grupo_con_etiquetas)}")
        else:
            top_vars = cat_vars[:top_n_variables]
        
        patrones_combinados = []
        n_clusters = final_df[cluster_col].nunique()
        
        print(f"\n  Analizando combinaciones entre las {top_n_variables} variables...")
        print(f"  Total de combinaciones posibles: {(top_n_variables*(top_n_variables-1))//2} pares")
        print(f"  Se evaluarán {n_clusters} clusters")
        
        print(f"  Calculando asociaciones globales...")
        cramers_global = {}
        
        for i in range(len(top_vars)):
            for j in range(i+1, len(top_vars)):
                var1, var2 = top_vars[i], top_vars[j]
                if var1 in final_df.columns and var2 in final_df.columns:
                    try:
                        data1 = final_df[var1].copy()
                        data2 = final_df[var2].copy()
                        
                        if pd.api.types.is_numeric_dtype(data1):
                            data1 = pd.qcut(data1, 4, duplicates='drop')
                        else:
                            data1 = data1.astype(str)
                            
                        if pd.api.types.is_numeric_dtype(data2):
                            data2 = pd.qcut(data2, 4, duplicates='drop')
                        else:
                            data2 = data2.astype(str)
                        
                        cross_table = pd.crosstab(data1, data2)
                        chi2, p, dof, expected = chi2_contingency(cross_table)
                        n = cross_table.sum().sum()
                        min_dim = min(cross_table.shape) - 1
                        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                        cramers_global[(var1, var2)] = cramers_v
                    except:
                        cramers_global[(var1, var2)] = 0
        
        print(f"  Analizando patrones por cluster...")
        
        for cluster_id in sorted(final_df[cluster_col].unique()):
            cluster_data = final_df[final_df[cluster_col] == cluster_id]
            cluster_size = len(cluster_data)
            
            if cluster_size < 20:
                continue
                
            print(f"    Cluster {cluster_id} ({cluster_size} observaciones)...")
            
            for i in range(len(top_vars)):
                for j in range(i+1, len(top_vars)):
                    var1, var2 = top_vars[i], top_vars[j]
                    
                    if var1 not in cluster_data.columns or var2 not in cluster_data.columns:
                        continue
                    
                    try:
                        data1_cluster = cluster_data[var1].copy()
                        data2_cluster = cluster_data[var2].copy()
                        
                        if pd.api.types.is_numeric_dtype(data1_cluster):
                            data1_cluster = pd.qcut(data1_cluster, 4, duplicates='drop')
                        else:
                            data1_cluster = data1_cluster.astype(str)
                            
                        if pd.api.types.is_numeric_dtype(data2_cluster):
                            data2_cluster = pd.qcut(data2_cluster, 4, duplicates='drop')
                        else:
                            data2_cluster = data2_cluster.astype(str)
                        
                        cluster_cross = pd.crosstab(data1_cluster, data2_cluster)
                        
                        if cluster_cross.size == 0:
                            continue
                        
                        chi2_cluster, p_cluster, dof_cluster, expected_cluster = chi2_contingency(cluster_cross)
                        n_cluster = cluster_cross.sum().sum()
                        min_dim_cluster = min(cluster_cross.shape) - 1
                        
                        if min_dim_cluster == 0:
                            continue
                            
                        cramers_v_cluster = np.sqrt(chi2_cluster / (n_cluster * min_dim_cluster))
                        
                        cramers_v_global = cramers_global.get((var1, var2), 0)
                        diferencia_abs = abs(cramers_v_cluster - cramers_v_global)
                        
                        if diferencia_abs > umbral_asociacion:
                            if cramers_v_cluster > cramers_v_global:
                                tipo = "MÁS FUERTE"
                            else:
                                tipo = "MÁS DÉBIL"
                            
                            if cramers_v_cluster >= 0.5:
                                fuerza_cluster = "FUERTE"
                            elif cramers_v_cluster >= 0.3:
                                fuerza_cluster = "MODERADA"
                            elif cramers_v_cluster >= 0.1:
                                fuerza_cluster = "DÉBIL"
                            else:
                                fuerza_cluster = "MUY DÉBIL"
                            
                            etiqueta1 = obtener_etiqueta(var1) or var1
                            etiqueta2 = obtener_etiqueta(var2) or var2
                            
                            patrones_combinados.append({
                                'Variables': f"{var1} × {var2}",
                                'Etiquetas': f"{etiqueta1[:30]} × {etiqueta2[:30]}",
                                'Cluster': cluster_id,
                                'Tamaño_Cluster': cluster_size,
                                'Asociacion_Global': cramers_v_global,
                                'Asociacion_Cluster': cramers_v_cluster,
                                'Diferencia': cramers_v_cluster - cramers_v_global,
                                'Diferencia_Abs': diferencia_abs,
                                'Tipo_Cambio': tipo,
                                'Fuerza_Cluster': fuerza_cluster,
                                'p_valor': p_cluster
                            })
                            
                    except Exception:
                        continue
        
        if not patrones_combinados:
            print(f"\n  No se encontraron patrones de interacción significativos.")
            resultados_patrones[cluster_col] = []
            continue
        
        print(f"\n  RESUMEN: Se encontraron {len(patrones_combinados)} patrones de interacción significativos")
        
        clusters_unicos = sorted(set([p['Cluster'] for p in patrones_combinados]))
        
        for cluster_id in clusters_unicos:
            patrones_cluster = [p for p in patrones_combinados if p['Cluster'] == cluster_id]
            patrones_cluster.sort(key=lambda x: x['Diferencia_Abs'], reverse=True)
            top_n = min(10, len(patrones_cluster))
            
            print(f"\n  --- CLUSTER {cluster_id} ---")
            print(f"  {len(patrones_cluster)} patrones de interacción significativos encontrados")
            print(f"  (Mostrando los {top_n} más fuertes)")
            print(f"  {'─' * 80}")
            
            for idx, patron in enumerate(patrones_cluster[:top_n], 1):
                vars_con_etiquetas = []
                vars_list = patron['Variables'].split(' × ')
                for v in vars_list:
                    etiqueta = obtener_etiqueta(v)
                    if etiqueta:
                        vars_con_etiquetas.append(f"{v} ({etiqueta[:30]})")
                    else:
                        vars_con_etiquetas.append(v)
                
                var_names = ' × '.join(vars_con_etiquetas)
                if len(var_names) > 80:
                    var_names = var_names[:77] + "..."
                
                simbolo = "↑" if patron['Tipo_Cambio'] == "MÁS FUERTE" else "↓"
                
                print(f"  {idx:2d}. {simbolo} {var_names}")
                print(f"       Asociación: Cluster={patron['Asociacion_Cluster']:.3f} ({patron['Fuerza_Cluster']}) "
                      f"vs Global={patron['Asociacion_Global']:.3f}")
                print(f"       Diferencia: {patron['Diferencia']:+.3f} ({patron['Tipo_Cambio']})")
                print(f"       p-valor: {patron['p_valor']:.4f}")
                print()
        
        resultados_patrones[cluster_col] = patrones_combinados
    
    print("\n" + "=" * 80)
    print("FIN DEL ANÁLISIS DE PATRONES COMBINADOS")
    print("=" * 80)
    
    return resultados_patrones


def analizar_jerarquia_clusters(final_df, cluster_cols):
    print("\n" + "=" * 80)
    print("5. ANÁLISIS DE JERARQUÍA Y ESTABILIDAD ENTRE AGRUPACIONES")
    print("=" * 80)
    
    resultados_jerarquia = {}
    
    if len(cluster_cols) > 1:
        print("\nEvolución jerárquica de las clases:")
        
        for i in range(len(cluster_cols) - 1):
            col_actual = cluster_cols[i]
            col_siguiente = cluster_cols[i + 1]
            
            print(f"\n  {col_actual} → {col_siguiente}:")
            print("  " + "-" * 60)
            
            tabla_conteo = pd.crosstab(final_df[col_actual], final_df[col_siguiente])
            transicion_porcentaje = pd.crosstab(final_df[col_actual], 
                                               final_df[col_siguiente], 
                                               normalize='index')
            
            for clase_actual in sorted(final_df[col_actual].unique()):
                total_obs = tabla_conteo.loc[clase_actual].sum()
                distribucion = transicion_porcentaje.loc[clase_actual]
                distribucion_ordenada = distribucion.sort_values(ascending=False)
                distribucion_filtrada = distribucion_ordenada[distribucion_ordenada > 0]
                
                transiciones = []
                total_acumulado = 0
                
                for clase_dest, porcentaje in distribucion_filtrada.items():
                    frecuencia = tabla_conteo.loc[clase_actual, clase_dest]
                    transicion_str = f"{porcentaje:.1%} → {clase_dest} ({frecuencia} obs)"
                    transiciones.append(transicion_str)
                    total_acumulado += porcentaje
                    if total_acumulado >= 0.995 or len(transiciones) >= 5:
                        break
                
                if total_acumulado < 0.995 and len(distribucion_filtrada) > len(transiciones):
                    porcentaje_restante = 1 - total_acumulado
                    clusters_restantes = distribucion_filtrada.index[len(transiciones):]
                    frecuencia_restante = sum([tabla_conteo.loc[clase_actual, dest] 
                                              for dest in clusters_restantes])
                    if porcentaje_restante > 0.001:
                        transiciones.append(f"{porcentaje_restante:.1%} → otros ({frecuencia_restante} obs)")
                
                print(f"  Clase {clase_actual} (total={total_obs} obs): {', '.join(transiciones)}")
            
            print("  " + "-" * 60)
            
            print("\n  Resumen tabular:")
            print("  Clase | Total obs | Distribución principal")
            print("  " + "-" * 50)
            
            for clase_actual in sorted(final_df[col_actual].unique()):
                total_obs = tabla_conteo.loc[clase_actual].sum()
                distribucion = transicion_porcentaje.loc[clase_actual]
                principales = distribucion.nlargest(2)
                
                if len(principales) == 2:
                    dest1, pct1 = principales.index[0], principales.iloc[0]
                    dest2, pct2 = principales.index[1], principales.iloc[1]
                    obs1 = tabla_conteo.loc[clase_actual, dest1]
                    obs2 = tabla_conteo.loc[clase_actual, dest2]
                    print(f"  {clase_actual:^5} | {total_obs:^9} | {pct1:.1%}→{dest1}({obs1}), {pct2:.1%}→{dest2}({obs2})")
                elif len(principales) == 1:
                    dest, pct = principales.index[0], principales.iloc[0]
                    obs = tabla_conteo.loc[clase_actual, dest]
                    print(f"  {clase_actual:^5} | {total_obs:^9} | {pct:.1%}→{dest}({obs})")
            
            resultados_jerarquia[f"{col_actual}_{col_siguiente}"] = {
                'tabla_conteo': tabla_conteo,
                'transicion_porcentaje': transicion_porcentaje
            }
    
    return resultados_jerarquia


def generar_sintesis_final(final_df, cluster_cols, cat_vars, variables_discriminantes_global):
    print("\n" + "=" * 80)
    print("6. SÍNTESIS INTEGRAL (CON TODAS LAS VARIABLES)")
    print("=" * 80)
    
    evaluacion_final = []
    
    for cluster_col in cluster_cols:
        n_clusters = final_df[cluster_col].nunique()
        cluster_sizes = final_df[cluster_col].value_counts().values
        
        size_std = cluster_sizes.std()
        size_mean = cluster_sizes.mean()
        balance_score = 1 - (size_std / size_mean) if size_mean > 0 else 0
        
        if cluster_col in variables_discriminantes_global:
            discriminacion_scores = variables_discriminantes_global[cluster_col]['Cramers_V'].head(20)
            discriminacion_score = discriminacion_scores.mean() if len(discriminacion_scores) > 0 else 0
        else:
            discriminacion_score = 0
        
        coherencia_scores = []
        vars_coherencia = cat_vars[:10] if len(cat_vars) >= 10 else cat_vars
        for cluster_id in final_df[cluster_col].unique():
            cluster_data = final_df[final_df[cluster_col] == cluster_id]
            for var in vars_coherencia:
                if var in cluster_data.columns:
                    var_data = cluster_data[var].copy()
                    if pd.api.types.is_numeric_dtype(var_data):
                        try:
                            var_data = pd.qcut(var_data, 4, duplicates='drop')
                        except:
                            var_data = var_data.astype(str)
                    else:
                        var_data = var_data.astype(str)
                    frecuencias = var_data.value_counts(normalize=True)
                    if not frecuencias.empty:
                        max_freq = frecuencias.iloc[0]
                        coherencia_scores.append(max_freq)
        
        coherencia_score = np.mean(coherencia_scores) if coherencia_scores else 0
        
        puntuacion_combinada = (balance_score * 0.25 + 
                               discriminacion_score * 0.5 + 
                               coherencia_score * 0.25)
        
        if puntuacion_combinada > 0.7:
            recomendacion = "EXCELENTE"
        elif puntuacion_combinada > 0.5:
            recomendacion = "BUENA"
        elif puntuacion_combinada > 0.3:
            recomendacion = "ACEPTABLE"
        else:
            recomendacion = "LIMITADA"
        
        evaluacion_final.append({
            'Agrupacion': cluster_col,
            'N_Clases': n_clusters,
            'Balance': f"{balance_score:.3f}",
            'Discriminacion': f"{discriminacion_score:.3f}",
            'Coherencia': f"{coherencia_score:.3f}",
            'Puntuacion_Total': f"{puntuacion_combinada:.3f}",
            'Recomendacion': recomendacion
        })
    
    eval_df = pd.DataFrame(evaluacion_final)
    print("\nEVALUACIÓN COMPARATIVA DE TODAS LAS AGRUPACIONES:")
    print("-" * 90)
    print(eval_df.sort_values('Puntuacion_Total', ascending=False).to_string(index=False))
    print("-" * 90)
    
    return eval_df


def _crear_tabla_consolidada(lista_caracteristicas):
    if not lista_caracteristicas:
        return None
    
    data = [['Variable', 'Categoría', 'Freq Clase', 'Abs Clase', 'Freq Global', 'Abs Global', 'Dif pp']]
    for item in lista_caracteristicas:
        var = item['Variable'][:30] + ('...' if len(item['Variable']) > 30 else '')
        cat = item['Categoria'][:40] + ('...' if len(item['Categoria']) > 40 else '')
        data.append([
            var,
            cat,
            f"{item['Freq_Cluster_Rel']:.1%}",
            str(item['Count_Cluster']),
            f"{item['Freq_Global_Rel']:.1%}",
            str(item['Count_Global']),
            f"{item['Diferencia']*100:+.1f}pp"
        ])
    
    tabla = Table(data, repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    return tabla

def graficar_arbol_jerarquico(transiciones_dict, cluster_cols, final_df, figsize=(12, 8)):
    if len(cluster_cols) < 2:
        return None
    
    sizes = {}
    for col in cluster_cols:
        sizes[col] = final_df[col].value_counts().to_dict()
    
    niveles = {col: i for i, col in enumerate(cluster_cols)}
    num_niveles = len(cluster_cols)
    
    posiciones = {}
    for col in cluster_cols:
        clases = sorted(sizes[col].keys())
        n_clases = len(clases)
        for idx, cl in enumerate(clases):
            y = (idx + 0.5) / n_clases
            posiciones[(col, cl)] = (niveles[col], y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    max_flow = 0
    edges = []
    for key, dict_trans in transiciones_dict.items():
        partes = key.split('_')
        col_orig = '_'.join(partes[:2])
        col_dest = '_'.join(partes[2:])
        tabla_conteo = dict_trans['tabla_conteo']
        
        for clase_orig in tabla_conteo.index:
            for clase_dest in tabla_conteo.columns:
                flujo = tabla_conteo.loc[clase_orig, clase_dest]
                if flujo > 0:
                    edges.append({
                        'orig': (col_orig, clase_orig),
                        'dest': (col_dest, clase_dest),
                        'flujo': flujo
                    })
                    if flujo > max_flow:
                        max_flow = flujo
    
    edges.sort(key=lambda x: x['flujo'], reverse=True)
    for e in edges:
        x0, y0 = posiciones[e['orig']]
        x1, y1 = posiciones[e['dest']]
        linewidth = 0.5 + 5 * (e['flujo'] / max_flow)
        ax.plot([x0, x1], [y0, y1], 'b-', linewidth=linewidth, alpha=0.6, solid_capstyle='round')
    
    for col in cluster_cols:
        for cl, size in sizes[col].items():
            x, y = posiciones[(col, cl)]
            node_size = 200 + 1000 * (size / final_df.shape[0])
            ax.scatter(x, y, s=node_size, c='red', edgecolors='black', zorder=5, alpha=0.9)
            ax.text(x, y, f"{cl}\n({size})", ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)
    
    ax.set_xlim(-0.5, num_niveles - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(num_niveles))
    ax.set_xticklabels(cluster_cols, fontsize=10)
    ax.set_yticks([])
    ax.set_title("Jerarquía de clases a través de los distintos k", fontsize=14)
    ax.set_xlabel("Número de clases (k)", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    return Image(buf, width=figsize[0]*inch, height=figsize[1]*inch)


def generar_dendrograma_clases(final_df, cat_vars, cluster_col, variables_clave=None, top_n=50, figsize=(10,6)):
    if not SCIPY_AVAILABLE:
        return None
    
    clases = sorted(final_df[cluster_col].unique())
    n_clases = len(clases)
    if n_clases < 2:
        return None
    
    if variables_clave is not None and cluster_col in variables_clave:
        df_vars = variables_clave[cluster_col]
        if not df_vars.empty:
            top_vars = df_vars.head(top_n)['Variable'].tolist()
        else:
            top_vars = cat_vars[:top_n]
    else:
        top_vars = cat_vars[:top_n]
    
    categorias_por_var = {}
    for var in top_vars:
        if var in final_df.columns:
            cats = final_df[var].dropna().unique()
            cats = [str(c) for c in cats]
            categorias_por_var[var] = sorted(cats)
    
    if not categorias_por_var:
        return None
    
    all_cats = []
    for var, cats in categorias_por_var.items():
        for cat in cats:
            all_cats.append(f"{var}::{cat}")
    
    perfil_matrix = np.zeros((n_clases, len(all_cats)))
    for i, clase in enumerate(clases):
        datos_clase = final_df[final_df[cluster_col] == clase]
        total_clase = len(datos_clase)
        for j, cat_key in enumerate(all_cats):
            var, cat = cat_key.split("::")
            if var in datos_clase.columns:
                freq = (datos_clase[var].astype(str) == cat).sum() / total_clase if total_clase > 0 else 0
                perfil_matrix[i, j] = freq
    
    dist_matrix = pdist(perfil_matrix, metric='euclidean')
    linkage_matrix = linkage(dist_matrix, method='ward')
    
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(linkage_matrix, labels=[f"Clase {c}" for c in clases], ax=ax, leaf_rotation=90)
    ax.set_title(f"Dendrograma de Clases para {cluster_col} (basado en top {len(top_vars)} variables)")
    ax.set_xlabel("Clases")
    ax.set_ylabel("Distancia")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    return Image(buf, width=figsize[0]*inch, height=figsize[1]*inch)


def _crear_tabla_completitud(final_df, cat_vars):
    data = [['Variable', 'No Nulos', 'Total', 'Completitud (%)']]
    total_obs = len(final_df)
    for var in cat_vars:
        if var in final_df.columns:
            n_non_null = final_df[var].count()
            pct = (n_non_null / total_obs) * 100
            data.append([
                var,
                str(n_non_null),
                str(total_obs),
                f"{pct:.1f}%"
            ])
    tabla = Table(data, repeatRows=1)
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    return tabla

def generar_reporte_pdf(resultados, filename, final_df, cat_vars, cluster_cols, obtener_etiqueta=None):
    if not isinstance(cluster_cols, (list, tuple, pd.Series, np.ndarray)):
        try:
            cluster_cols = list(cluster_cols)
        except:
            cluster_cols = []
    else:
        cluster_cols = list(cluster_cols)

    doc = SimpleDocTemplate(filename, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = styles['Title']
    story.append(Paragraph("Análisis Estadístico de Clases", title_style))
    story.append(Spacer(1, 12))
    
    if 'metrics' in resultados:
        story.append(Paragraph("0. Métricas de los Modelos", styles['Heading2']))
        df_metrics = resultados['metrics']
        data = [df_metrics.columns.tolist()] + df_metrics.values.tolist()
        tabla = Table(data, repeatRows=1)
        tabla.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(tabla)
        story.append(Spacer(1, 12))
    
    story.append(Paragraph("1. Distribución de Clases", styles['Heading2']))
    if 'distribucion' in resultados:
        df_dist = resultados['distribucion']
        data = [df_dist.columns.tolist()] + df_dist.values.tolist()
        tabla = Table(data, repeatRows=1)
        tabla.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(tabla)
        story.append(Spacer(1, 12))
    
    story.append(Paragraph("2. Tablas completas de Cramér's V por agrupación", styles['Heading2']))
    if 'cramers_completos' in resultados:
        for k, df_cramers in resultados['cramers_completos'].items():
            if not df_cramers.empty:
                story.append(Paragraph(f"Agrupación: {k}", styles['Heading3']))
                df_mostrar = df_cramers.head(50)
                data = [df_mostrar.columns.tolist()] + df_mostrar.values.tolist()
                tabla = Table(data, repeatRows=1)
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 8),
                    ('BOTTOMPADDING', (0,0), (-1,0), 6),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(tabla)
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(f"Agrupación {k}: No hay datos", styles['Normal']))
    
    story.append(Paragraph("3. Variables más discriminantes (Top 25 por agrupación)", styles['Heading2']))
    if 'variables_clave' in resultados:
        for k, df_vars in resultados['variables_clave'].items():
            if not df_vars.empty:
                story.append(Paragraph(f"Agrupación: {k}", styles['Heading3']))
                df_top = df_vars.head(25)
                data = [['Variable', "Cramér's V", 'p-value', 'χ²']]
                for _, row in df_top.iterrows():
                    data.append([
                        row['Variable'],
                        f"{row['Cramers_V']:.4f}",
                        f"{row['p_value']:.2e}",
                        f"{row['Chi2']:.0f}"
                    ])
                tabla = Table(data, repeatRows=1)
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 8),
                    ('BOTTOMPADDING', (0,0), (-1,0), 6),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black)
                ]))
                story.append(tabla)
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(f"Agrupación {k}: No hay variables significativas", styles['Normal']))
    
    story.append(Paragraph("4. Perfiles de Clases (detalle)", styles['Heading2']))
    if 'perfiles' in resultados:
        for k, perfiles in resultados['perfiles'].items():
            story.append(Paragraph(f"Agrupación: {k}", styles['Heading3']))
            for perfil in perfiles:
                story.append(Paragraph(f"Clase {perfil['Clase']}: {perfil['Tamaño']} obs ({perfil['Porcentaje']:.1f}%)", styles['Normal']))
                
                if not perfil['Sobre'] and not perfil['Sub']:
                    story.append(Paragraph("  No hay variables con diferencias significativas.", styles['Normal']))
                else:
                    if perfil['Sobre']:
                        story.append(Paragraph("  SOBRERREPRESENTADAS:", styles['Normal']))
                        tabla_sobre = _crear_tabla_consolidada(perfil['Sobre'])
                        if tabla_sobre:
                            story.append(tabla_sobre)
                            story.append(Spacer(1, 6))
                    
                    if perfil['Sub']:
                        story.append(Paragraph("  SUBREPRESENTADAS:", styles['Normal']))
                        tabla_sub = _crear_tabla_consolidada(perfil['Sub'])
                        if tabla_sub:
                            story.append(tabla_sub)
                            story.append(Spacer(1, 6))
                story.append(Spacer(1, 12))
    
    story.append(Paragraph("5. Flujo de clases entre agrupaciones", styles['Heading2']))
    if 'jerarquia' in resultados and len(cluster_cols) >= 2:
        img_sankey = generar_sankey_flujo_clases(resultados['jerarquia'], cluster_cols, final_df, return_image=True)
        if img_sankey:
            story.append(img_sankey)
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph("No se pudo generar el diagrama de Sankey. Se muestran heatmaps alternativos.", styles['Normal']))
            for i in range(len(cluster_cols) - 1):
                col_orig = cluster_cols[i]
                col_dest = cluster_cols[i + 1]
                key = f"{col_orig}_{col_dest}"
                if key not in resultados['jerarquia']:
                    continue
                tabla_conteo = resultados['jerarquia'][key]['tabla_conteo']
                fig = graficar_matriz_transicion(tabla_conteo, col_orig, col_dest)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                plt.close(fig)
                buf.seek(0)
                img = Image(buf, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))

        story.append(Paragraph("Detalle numérico de transiciones:", styles['Heading3']))
        tablas_flujo = crear_tablas_flujo(resultados['jerarquia'], cluster_cols)
        for titulo, tabla in tablas_flujo:
            story.append(Paragraph(titulo, styles['Normal']))
            story.append(tabla)
            story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No hay datos de jerarquía para generar flujo.", styles['Normal']))
    
    story.append(Paragraph("6. Anexo: Etiquetas de Variables", styles['Heading2']))
    if obtener_etiqueta:
        tabla_etiquetas = _crear_tabla_etiquetas(final_df, cat_vars, obtener_etiqueta)
        if tabla_etiquetas:
            story.append(tabla_etiquetas)
        else:
            story.append(Paragraph("No hay variables para mostrar.", styles['Normal']))
    else:
        story.append(Paragraph("No se proporcionó un mapeo de etiquetas.", styles['Normal']))
    story.append(Spacer(1, 12))
    
    doc.build(story)
    print(f"Reporte PDF guardado como: {filename}")

def graficar_matriz_transicion(tabla_conteo, col_orig, col_dest, figsize=(6,4)):
    tabla_porc = tabla_conteo.div(tabla_conteo.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(tabla_porc, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(tabla_porc.columns)))
    ax.set_yticks(np.arange(len(tabla_porc.index)))
    ax.set_xticklabels(tabla_porc.columns)
    ax.set_yticklabels(tabla_porc.index)
    ax.set_xlabel(f'Clases {col_dest}')
    ax.set_ylabel(f'Clases {col_orig}')
    ax.set_title(f'Transición {col_orig} → {col_dest} (% fila)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(tabla_porc.index)):
        for j in range(len(tabla_porc.columns)):
            text = ax.text(j, i, f"{tabla_porc.iloc[i, j]:.1f}%",
                           ha="center", va="center",
                           color="w" if tabla_porc.iloc[i, j] > 50 else "black")
    
    fig.colorbar(im, ax=ax, label='Porcentaje')
    plt.tight_layout()
    return fig

def asignar_letras_estables(df, cluster_cols, umbral=0.8):
    df_result = df.copy()
    mapeos = {}
    
    col0 = cluster_cols[0]
    freq0 = df[col0].value_counts().sort_values(ascending=False)
    letras = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    mapeo0 = {}
    for val in freq0.index:
        mapeo0[val] = next(letras)
    df_result[col0] = df_result[col0].map(mapeo0)
    mapeos[col0] = mapeo0
    
    for i in range(1, len(cluster_cols)):
        col_prev = cluster_cols[i-1]
        col_curr = cluster_cols[i]
        
        tabla = pd.crosstab(df[col_prev], df[col_curr])
        tabla_pct = tabla.div(tabla.sum(axis=1), axis=0)
        
        herencias = {}
        for prev_val in tabla.index:
            destino_max = tabla_pct.loc[prev_val].idxmax()
            pct_max = tabla_pct.loc[prev_val, destino_max]
            if pct_max >= umbral:
                if destino_max in herencias:
                    otro_origen, otro_pct = herencias[destino_max]
                    if pct_max > otro_pct:
                        herencias[destino_max] = (prev_val, pct_max)
                else:
                    herencias[destino_max] = (prev_val, pct_max)
        
        mapeo_curr = {}
        letras_asignadas_curr = set()
        
        for dest_val, (orig_val, _) in herencias.items():
            letra_orig = mapeos[col_prev][orig_val]
            mapeo_curr[dest_val] = letra_orig
            letras_asignadas_curr.add(letra_orig)
        
        clases_curr = sorted(df[col_curr].unique())
        letras_usadas_global = set()
        for col in cluster_cols[:i]:
            letras_usadas_global.update(mapeos[col].values())
        letras_usadas_global.update(letras_asignadas_curr)
        
        for val in clases_curr:
            if val not in mapeo_curr:
                for letra in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    if letra not in letras_usadas_global:
                        nueva_letra = letra
                        break
                else:
                    nueva_letra = f"Z{len(letras_usadas_global)}"
                mapeo_curr[val] = nueva_letra
                letras_usadas_global.add(nueva_letra)
        
        df_result[col_curr] = df_result[col_curr].map(mapeo_curr)
        mapeos[col_curr] = mapeo_curr
    
    return df_result, mapeos