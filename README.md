# README – Latent Class Analysis (LCA) for Symptom Data

This repository contains a complete pipeline for performing **Latent Class Analysis (LCA)** on categorical DISC-IV symptom data using the **Expectation-Maximization (EM)** algorithm. It also includes extensive post‑analysis tools: class profiles, Cramér’s V associations, hierarchical transitions, and a detailed PDF report. An age‑filtering option is provided to analyse children (≤11 years) and teenagers separately.

## Requirements

- Python 3.8 or higher
- Required packages:
  - `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`
  - `plotly`, `kaleido` (for Sankey diagrams in PDF)
  - `reportlab` (for PDF generation)
  - `scikit-learn` (for the alternative Gaussian Mixture LCA – not used by default)
- Install all dependencies with:

```bash
pip install pandas numpy scipy matplotlib seaborn plotly kaleido reportlab scikit-learn
```

## Files Provided

| File | Description |
|------|-------------|
| `lca.py` | EM‑based LCA implementation (`LatentClassAnalysis` class) and main function `perform_true_lca` that runs the analysis for k=2..6. |
| `utils.py` | Helper functions for data preparation, code mapping, statistical analysis, and PDF report generation. |
| `Example.ipynb` | A Jupyter notebook that runs the complete workflow (see below for the code). |
| `labels.xlsx` | Excel file with variable codes (`Código`) and their human‑readable labels (`Etiqueta`). This is used to annotate the output. |

## Your Data

You need to provide the following datasets (CSV format):

- **`symptom_data.csv`** – contains the symptom variables. Must include an `id` column that matches the identifier in the base data. For this dataset, columns should be symptom/question codes (e.g. pad001, which corresponds to the label 'Trouble keeping mind on task for more than a short period of time', according to the provided label dataset) and individual answers should be numerically coded (0,2,3,7,8,9, etc.) following the computer-administered Spanish version of the Diagnostic Interview Schedule for Children Version IV (DISC-IV).
- **`base_data.csv`** – contains the same identifiers (`inum`) and an age column (`ed1`). Used to filter by age group.
- **`labels.xlsx`** – mapping of variable codes to labels (provided with this package).

Make sure the column names match those used in the code (you can adjust them in the script if needed).

## Workflow

The `Example.ipynb` notebook executes the following steps:

1. **Import libraries and modules** – loads all necessary Python packages and the custom functions from `lca.py` and `utils.py`.
2. **Load data** – reads your symptom data, label mapping, and base data (with age).
3. **Merge age information** – joins the base data onto the symptom data using the `id`/`inum` columns.
4. **Filter by age group** – selects either all observations, children (`ed1 ≤ 11`), or teenagers (`ed1 > 11`).
5. **Run LCA** – for k = 2 to 6, using the EM implementation from `lca.py`. Returns:
   - `output_df`: original data plus class assignment columns (`lca_k2` … `lca_k6`)
   - `prob_df`: membership probabilities for each class and each k
   - `metric_df`: model fit metrics (BIC, AIC, log‑likelihood, certainty, entropy)
6. **Generate comprehensive PDF report** – calls `analisis_estadistico_clusters` from `utils.py` to produce a detailed report containing:
   - Class distributions (size, completeness)
   - Cramér’s V between each variable and the class assignments (for all k)
   - Top discriminating variables (p < 0.001)
   - Class profiles (over‑/under‑represented categories)
   - Transition matrices between consecutive k values
   - Sankey diagram and hierarchical tree of class flows
7. **Compute pairwise Cramér’s V matrices** – calculates the association between every pair of original variables using `analisis_interacciones_variables` (from `utils.py`). Saves the results as CSV files.

All outputs are saved in the current working directory, with optional suffixes (`_child`, `_teen`) when age filtering is applied.

## How to Run

1. Place all provided files (`lca.py`, `utils.py`, `labels.xlsx`) in the same folder as your data.
2. Open `Example.ipynb` in Jupyter Notebook or copy the code block below into a Python script.
3. Edit the file paths in the script to point to your actual data files.
4. Choose the age group by setting `age_group = 'all'`, `'child'`, or `'teen'`.
5. Run the script (or the notebook cells).
6. After execution, you will find the generated output files in the same directory.

## Example Code Block

Below is the core script (the content of `Example.ipynb`). You can copy it directly into a `.py` file or run it cell‑by‑cell in a notebook.

```python
# =============================================================================
# COMPLETE LCA WORKFLOW WITH AGE FILTERING + PAIRWISE CRAMÉR'S V
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CONFIGURATION: SELECT AGE GROUP TO ANALYSE
# -----------------------------------------------------------------------------
age_group = 'all'          # Options: 'all', 'child', 'teen'
# 'child' = age ≤ 11, 'teen' = age > 11

# -----------------------------------------------------------------------------
# 2. IMPORT ALL REQUIRED LIBRARIES AND MODULES
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

from lca import (
    perform_true_lca,
    plot_lca_results,
    LatentClassAnalysis,
    prepare_data_for_lca,
    calculate_imc_categories,
    apply_code_map_without_imputation,
    CODE_MAP
)

from utils import (
    build_full_dict,
    question_list,
    analisis_estadistico_clusters,
    analisis_interacciones_variables
)

# -----------------------------------------------------------------------------
# 3. LOAD YOUR DATA
# -----------------------------------------------------------------------------
df_sintomas_disc = pd.read_csv('path/to/your/symptom_data.csv')
df_etiquetas = pd.read_csv('path/to/your/labels.csv')
df_base = pd.read_csv('path/to/your/base_data.csv')

print(f"Data shape: {df_sintomas_disc.shape}")
print(f"Label mapping shape: {df_etiquetas.shape}")
print(f"Base data shape: {df_base.shape}")

# -----------------------------------------------------------------------------
# 4. MERGE AGE INFORMATION AND FILTER BY AGE GROUP
# -----------------------------------------------------------------------------
df_merged = pd.merge(
    df_sintomas_disc,
    df_base[['inum', 'ed1']],
    left_on='id',
    right_on='inum',
    how='left'
).drop(columns=['inum'])

print(f"Data after merging age: {df_merged.shape}")

if age_group == 'child':
    df_work = df_merged[df_merged['ed1'] <= 11].copy()
    suffix = "_child"
elif age_group == 'teen':
    df_work = df_merged[df_merged['ed1'] > 11].copy()
    suffix = "_teen"
else:
    df_work = df_merged.copy()
    suffix = ""

print(f"\nWorking with {age_group.upper()} group: {df_work.shape[0]} observations")
print(f"Age range: {df_work['ed1'].min()} – {df_work['ed1'].max()}")

# -----------------------------------------------------------------------------
# 5. BUILD FULL DICTIONARY (optional)
# -----------------------------------------------------------------------------
full_dict = build_full_dict(df_work)
print("Full dictionary built successfully.")

# -----------------------------------------------------------------------------
# 6. GET THE LIST OF QUESTIONS FOR LCA
# -----------------------------------------------------------------------------
q_list = question_list()
print(f"Number of questions selected for LCA: {len(q_list)}")

# -----------------------------------------------------------------------------
# 7. PERFORM LCA FOR k = 2 TO 6 (EM ALGORITHM)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print(f"Running LCA with k = 2 to 6 for {age_group.upper()} group...")
print("="*70)

output_df, prob_df, metric_df = perform_true_lca(
    df=df_work,
    question_list=q_list,
    id_column='id',
    n_components_range=range(2, 7)
)

# -----------------------------------------------------------------------------
# 8. SAVE LCA OUTPUTS (with group suffix)
# -----------------------------------------------------------------------------
output_df.to_csv(f'output_df_lca{suffix}.csv', index=False)
prob_df.to_csv(f'prob_df_lca{suffix}.csv', index=False)
metric_df.to_csv(f'metric_df_lca{suffix}.csv', index=False)
print("\nLCA results saved to CSV files.")

# -----------------------------------------------------------------------------
# 9. GENERATE COMPREHENSIVE PDF REPORT (CLUSTER ANALYSIS)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print(f"Generating PDF report for {age_group.upper()} group...")
print("="*70)

resultados = analisis_estadistico_clusters(
    final_df=output_df,
    id_column='id',
    cluster_prefix='lca_k',
    df_etiquetas=df_etiquetas,
    col_codigo='Código',
    col_etiqueta='Etiqueta',
    output_pdf=f'reporte_lca{suffix}.pdf',
    metrics_df=metric_df,
    mostrar_todas_variables_perfil=True,
    max_variables_perfil=None
)

# =============================================================================
# 10. EXTRACT KEY RESULTS FROM THE 'resultados' DICTIONARY
# =============================================================================
metrics = resultados.get('metrics')
distribucion = resultados.get('distribucion')
cramers_all = resultados.get('cramers_completos')
perfiles = resultados.get('perfiles')
jerarquia = resultados.get('jerarquia')

# -----------------------------------------------------------------------------
# 11. COMPUTE PAIRWISE CRAMÉR'S V AMONG ORIGINAL VARIABLES (using utils function)
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("Computing pairwise Cramér's V matrices for all original variables")
print("="*70)

excluir = ['id'] + [col for col in output_df.columns if col.startswith('lca_k')]
cramersv_mat, pvalue_mat, chi2_mat = analisis_interacciones_variables(
    df=output_df,
    exclude_cols=excluir
)

cramersv_mat.to_csv(f'cramersv_pairwise_matrix{suffix}.csv')
pvalue_mat.to_csv(f'pvalue_pairwise_matrix{suffix}.csv')
chi2_mat.to_csv(f'chi2_pairwise_matrix{suffix}.csv')
print("Pairwise Cramér's V matrices saved to CSV files.")
```

## Output Files

| File | Description |
|------|-------------|
| `output_df_lca[_child|_teen].csv` | Original data plus class assignment columns (`lca_k2` … `lca_k6`). |
| `prob_df_lca[_child|_teen].csv` | Membership probabilities for each class and k. |
| `metric_df_lca[_child|_teen].csv` | Model fit metrics (BIC, AIC, log‑likelihood, iterations, convergence, certainty, entropy). |
| `reporte_lca[_child|_teen].pdf` | Comprehensive PDF report with all statistical analyses and visualisations. |
| `cramersv_pairwise_matrix[_child|_teen].csv` | Pairwise Cramér’s V matrix between all original variables. |
| `pvalue_pairwise_matrix[_child|_teen].csv` | Corresponding p‑values. |
| `chi2_pairwise_matrix[_child|_teen].csv` | χ² statistics. |

## Customisation

- **Age group**: change `age_group` to `'child'` or `'teen'` to analyse subgroups. Set to `'all'` to include everyone.
- **Number of classes**: modify the `n_components_range` in the call to `perform_true_lca`.
- **Parameters in `analisis_estadistico_clusters`**: you can adjust `umbral_diferencia` (default 0.10) to change the threshold for over‑/under‑representation, or `top_n_variables` (default 25) to limit the number of variables displayed.
- **PDF content**: the report includes many sections; if you want to exclude some, you can edit the `generar_reporte_pdf` function in `utils.py` (but this is advanced).

## Notes

- The LCA implementation uses **mode imputation** for missing values (replaces NaNs with the most frequent category per variable). This is done only during preparation for the LCA; the original data are preserved in `output_df`.
- The code maps numeric codes (0,2,3,7,8,9, etc.) to meaningful labels (NO, SI, AV, NR, NA, NS) using the `CODE_MAP` dictionary.
- The BMI columns (`pea001z`, `pea002k`, `pea003k`) are automatically detected and transformed into IMC categories (`underweight`, `normal weight`, `overweight`, `obese`) if present in the question list.
- The pairwise Cramér’s V matrix can be very large if you have many variables; the function `analisis_interacciones_variables` reduces the number of categories for any variable exceeding `max_categories` (default 10) by grouping low‑frequency categories into “Otras”.

## Troubleshooting

- **Missing columns**: If some variables in `question_list` are not found in your data, they will be skipped and a warning printed. Make sure your column names match exactly.
- **ReportLab/Plotly errors**: If PDF generation fails, ensure `reportlab` and `plotly` (with `kaleido`) are installed. You can also run the analysis without generating the PDF by setting `output_pdf=None`.


## License

This code is provided as‑is for research purposes. Please cite appropriately if you use it in your work.

## Directories

Other directories in this repository include the results discussed in my thesis work, using a full dataset of N=1558, composed of a child Dataset n=831 amd a Teen Dataset n=727.

* Cramers_lca: Contains the tables that describe discrimination metrics for each symptom related to classification for every value of k in each dataset studied.
* LCA_metric_n: Contains the tables that describe statistical metrics (e.g. BIC or AIC) for each LCA classification.
* LCA_n: Contains the tables that describe every individual and its classification alongside the de-coded symptom responses.
* LCA_prob_n: Contains the tables that describe the posterior probability of every individual associated to every class.
* chi2_matrix: Contains the chi2 matrices for every single symptom.
* composiciones: Contains the specific answers to every question.
* cramers_matrix: Contains the Cramer's V matrices for every single symptom.
* pvalue_matrix: Contains the p-value matrices for every single symptom.
