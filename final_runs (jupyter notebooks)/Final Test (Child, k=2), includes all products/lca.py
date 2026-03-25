import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import entropy


class LatentClassAnalysis:
    """
    Efficient EM implementation for Latent Class Analysis (LCA).
    Works directly with categorical indices instead of one-hot encoding.
    """

    def __init__(self, n_classes: int, max_iter: int = 500, tol: float = 1e-6,
                 n_init: int = 10, random_state: int = None,
                 verbose: bool = False):
        """
        Initializes the LCA model.

        Parameters
        ----------
        n_classes : int
            Number of latent classes
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Tolerance for convergence (change in log-likelihood)
        n_init : int
            Number of random initializations
        random_state : int
            Seed for reproducibility
        verbose : bool
            Show progress of the EM algorithm
        """
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.verbose = verbose

        self.class_probs_ = None
        self.conditional_probs_ = None
        self.responsibilities_ = None
        self.log_likelihood_ = None
        self.bic_ = None
        self.aic_ = None
        self.n_iter_ = None
        self.converged_ = False

        self.categories_per_var_ = None
        self.variable_names_ = None
        self.category_mappings_ = None

        if random_state is not None:
            np.random.seed(random_state)

    def _initialize_parameters(self, n_vars: int,
                               categories_per_var: List[int]) -> None:
        """Initializes model parameters randomly."""
        self.class_probs_ = np.random.dirichlet(np.ones(self.n_classes))

        self.conditional_probs_ = []
        for j in range(n_vars):
            n_cats = categories_per_var[j]
            cond_probs_var = np.random.dirichlet(
                np.ones(n_cats),
                size=self.n_classes
            )
            self.conditional_probs_.append(cond_probs_var)

    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Computes the complete log‑likelihood of the model.

        Parameters
        ----------
        X : np.ndarray
            Matrix of categorical indices (n_samples, n_vars)
        """
        n_samples, n_vars = X.shape
        log_lik = 0.0

        for i in range(n_samples):
            class_terms = np.log(self.class_probs_ + 1e-10)

            for k in range(self.n_classes):
                for j in range(n_vars):
                    cat_idx = X[i, j]
                    class_terms[k] += np.log(
                        self.conditional_probs_[j][k, cat_idx] + 1e-10
                    )

            log_lik += logsumexp(class_terms)

        return log_lik

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E‑step: computes posterior probabilities γ_ik.

        Parameters
        ----------
        X : np.ndarray
            Matrix of categorical indices (n_samples, n_vars)
        """
        n_samples, n_vars = X.shape
        log_responsibilities = np.zeros((n_samples, self.n_classes))

        log_class_probs = np.log(self.class_probs_ + 1e-10)
        log_conditional_probs = [
            np.log(cond_probs + 1e-10) for cond_probs in self.conditional_probs_
        ]

        for i in range(n_samples):
            for k in range(self.n_classes):
                log_resp = log_class_probs[k]

                for j in range(n_vars):
                    cat_idx = X[i, j]
                    log_resp += log_conditional_probs[j][k, cat_idx]

                log_responsibilities[i, k] = log_resp

        log_sum = logsumexp(log_responsibilities, axis=1, keepdims=True)
        log_responsibilities -= log_sum

        responsibilities = np.exp(log_responsibilities)

        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        M‑step: updates parameters using posterior probabilities.

        Parameters
        ----------
        X : np.ndarray
            Matrix of categorical indices (n_samples, n_vars)
        responsibilities : np.ndarray
            Posterior probabilities γ_ik
        """
        n_samples, n_vars = X.shape

        self.class_probs_ = responsibilities.sum(axis=0) / n_samples

        for j in range(n_vars):
            n_cats = self.conditional_probs_[j].shape[1]

            for k in range(self.n_classes):
                numerator = np.zeros(n_cats)

                for c in range(n_cats):
                    mask = (X[:, j] == c)
                    numerator[c] = np.sum(responsibilities[mask, k])

                denominator = responsibilities[:, k].sum()

                if denominator > 0:
                    self.conditional_probs_[j][k, :] = numerator / denominator
                else:
                    self.conditional_probs_[j][k, :] = 1.0 / n_cats

    def fit(self, X: np.ndarray,
            categories_per_var: List[int],
            variable_names: List[str] = None,
            category_mappings: List[Dict[int, Any]] = None) -> 'LatentClassAnalysis':
        """
        Fits the LCA model to the data.

        Parameters
        ----------
        X : np.ndarray
            Matrix of categorical indices (n_samples, n_vars)
        categories_per_var : List[int]
            Number of categories per variable
        variable_names : List[str]
            Names of the variables
        category_mappings : List[Dict[int, Any]]
            Mapping from indices to original categories

        Returns
        -------
        self : fitted model
        """
        n_samples, n_vars = X.shape

        self.variable_names_ = variable_names or [f"V{j}" for j in range(n_vars)]
        self.categories_per_var_ = categories_per_var
        self.category_mappings_ = category_mappings

        best_log_lik = -np.inf
        best_params = None

        for init in range(self.n_init):
            if self.verbose and self.n_init > 1:
                print(f"  Initialization {init + 1}/{self.n_init}")

            self._initialize_parameters(n_vars, categories_per_var)

            prev_log_lik = -np.inf
            converged = False

            for iteration in range(self.max_iter):
                responsibilities = self._e_step(X)

                self._m_step(X, responsibilities)

                log_lik = self._compute_log_likelihood(X)

                log_lik_change = log_lik - prev_log_lik

                if self.verbose and iteration % 50 == 0:
                    print(f"    Iter {iteration}: log-lik = {log_lik:.2f}, "
                          f"Δ = {log_lik_change:.6f}")

                if abs(log_lik_change) < self.tol and iteration > 10:
                    converged = True
                    if self.verbose:
                        print(f"    Convergence at iteration {iteration}")
                    break

                prev_log_lik = log_lik

            if log_lik > best_log_lik:
                best_log_lik = log_lik
                best_params = {
                    'class_probs': self.class_probs_.copy(),
                    'conditional_probs': [p.copy() for p in self.conditional_probs_],
                    'responsibilities': responsibilities.copy(),
                    'log_likelihood': log_lik,
                    'n_iter': iteration + 1,
                    'converged': converged
                }

        self.class_probs_ = best_params['class_probs']
        self.conditional_probs_ = best_params['conditional_probs']
        self.responsibilities_ = best_params['responsibilities']
        self.log_likelihood_ = best_params['log_likelihood']
        self.n_iter_ = best_params['n_iter']
        self.converged_ = best_params['converged']

        self._compute_information_criteria(n_samples, n_vars, categories_per_var)

        return self

    def _compute_information_criteria(self, n_samples: int,
                                      n_vars: int,
                                      categories_per_var: List[int]) -> None:
        """Computes BIC and AIC."""
        n_parameters = (self.n_classes - 1)

        for j in range(n_vars):
            n_cats = categories_per_var[j]
            n_parameters += self.n_classes * (n_cats - 1)

        self.bic_ = -2 * self.log_likelihood_ + n_parameters * np.log(n_samples)
        self.aic_ = -2 * self.log_likelihood_ + 2 * n_parameters

    def predict_proba(self) -> np.ndarray:
        """Returns posterior probabilities."""
        if self.responsibilities_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.responsibilities_

    def predict(self) -> np.ndarray:
        """Returns class assignments (class with highest probability)."""
        return np.argmax(self.predict_proba(), axis=1)

    def get_class_profiles(self) -> Dict[str, np.ndarray]:
        """Returns class profiles (conditional probabilities)."""
        profiles = {}

        for k in range(self.n_classes):
            class_profile = []
            for j, cond_probs in enumerate(self.conditional_probs_):
                class_profile.append(cond_probs[k])

            profiles[f'Class_{k}'] = np.array(class_profile)

        return profiles

    def get_class_profiles_human_readable(self) -> Dict[str, Dict[str, List[Tuple[Any, float]]]]:
        """
        Returns class profiles in human‑readable format.

        Returns
        -------
        Dict with:
          - Key: class name
          - Value: dict with:
            - Key: variable name
            - Value: list of (category, probability)
        """
        if self.category_mappings_ is None:
            raise ValueError("No category mapping available.")

        profiles = {}

        for k in range(self.n_classes):
            class_profile = {}

            for j, cond_probs in enumerate(self.conditional_probs_):
                var_name = self.variable_names_[j] if self.variable_names_ else f"V{j}"
                category_mapping = self.category_mappings_[j]

                probs = cond_probs[k]

                category_probs = []
                for cat_idx, prob in enumerate(probs):
                    category = category_mapping.get(cat_idx, f"Cat_{cat_idx}")
                    category_probs.append((category, float(prob)))

                category_probs.sort(key=lambda x: x[1], reverse=True)
                class_profile[var_name] = category_probs

            profiles[f'Class_{k}'] = class_profile

        return profiles

    def get_entropy(self) -> float:
        """Computes entropy of class assignments."""
        if self.responsibilities_ is None:
            return 0.0

        sample_entropies = entropy(self.responsibilities_.T)
        return np.mean(sample_entropies)

    def get_average_certainty(self) -> float:
        """Computes average assignment certainty."""
        if self.responsibilities_ is None:
            return 0.0

        max_probs = self.responsibilities_.max(axis=1)
        return float(max_probs.mean())


def prepare_data_for_lca(df: pd.DataFrame, question_list: List[str]) -> Tuple:
    """
    Prepares data for LCA using categorical indices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data
    question_list : List[str]
        List of columns/variables to include

    Returns
    -------
    X : np.ndarray
        Matrix of categorical indices (n_samples, n_vars)
    categories_per_var : List[int]
        Number of categories per variable
    variable_names : List[str]
        Names of the variables
    category_mappings : List[Dict[int, Any]]
        Mapping from indices to original categories
    """
    X_columns = []
    categories_per_var = []
    variable_names = []
    category_mappings = []

    for col in question_list:
        if col not in df.columns:
            print(f"  ⚠ Column {col} not found, skipping")
            continue

        col_data = df[col].copy()

        mode_values = col_data.dropna().mode()
        if not mode_values.empty:
            mode_value = str(mode_values.iloc[0])
        else:
            mode_value = 'MISSING'

        col_data_filled = col_data.fillna(mode_value).astype(str)

        unique_cats = sorted(col_data_filled.unique())
        cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
        idx_to_cat = {idx: cat for cat, idx in cat_to_idx.items()}

        X_col = col_data_filled.map(cat_to_idx).values

        X_columns.append(X_col)
        categories_per_var.append(len(unique_cats))
        variable_names.append(col)
        category_mappings.append(idx_to_cat)

    if not X_columns:
        raise ValueError("No valid columns for LCA analysis")

    X = np.column_stack(X_columns)

    return X, categories_per_var, variable_names, category_mappings


def perform_true_lca(df: pd.DataFrame, question_list: List[str],
                     id_column: str = 'id',
                     n_components_range: range = range(2, 7)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete implementation of classical LCA with EM algorithm (optimized version).

    Returns
    -------
    output_df : DataFrame with class assignments
    prob_df : DataFrame with membership probabilities
    metric_df : DataFrame with model metrics
    """
    warnings.filterwarnings('ignore')

    print("="*70)
    print("LATENT CLASS ANALYSIS (LCA) - EM ALGORITHM (OPTIMIZED)")
    print("="*70)

    work_df = df.copy()
    print(f"\n[1] Preparing data: {len(work_df)} observations")

    imc_required_cols = ['pea001z', 'pea002k', 'pea003k']
    imc_cols_in_question_list = [col for col in imc_required_cols if col in question_list]

    if imc_cols_in_question_list and all(col in work_df.columns for col in imc_required_cols):
        print(f"[2] Processing BMI columns...")

        work_df = calculate_imc_categories(work_df)

        for col in imc_required_cols:
            if col in question_list:
                question_list.remove(col)

        imc_new_cols = ['IMC_pea002', 'IMC_pea003']
        for new_col in imc_new_cols:
            if new_col in work_df.columns:
                question_list.append(new_col)

        print(f"    ✓ BMI categorized: {imc_new_cols}")

    print(f"\n[3] Applying code mapping...")

    existing_cols = [col for col in question_list if col in work_df.columns]
    missing_cols = [col for col in question_list if col not in work_df.columns]

    if missing_cols:
        print(f"    ⚠ Columns not found: {missing_cols}")
        question_list = existing_cols

    work_df = apply_code_map_without_imputation(work_df, question_list)
    print(f"    ✓ Mapping applied to {len(question_list)} columns")

    print(f"\n[4] Checking data types...")
    for col in question_list:
        if col in work_df.columns and work_df[col].dtype not in ['object', 'category']:
            work_df[col] = work_df[col].astype(str)

    print(f"\n[5] Missing value analysis...")
    missing_info = []
    for col in question_list:
        if col in work_df.columns:
            nan_count = work_df[col].isna().sum()
            if nan_count > 0:
                nan_percent = (nan_count / len(work_df)) * 100
                missing_info.append((col, nan_count, nan_percent))

    if missing_info:
        for col, nan_count, nan_percent in sorted(missing_info, key=lambda x: x[2], reverse=True):
            print(f"    {col:<15}: {nan_count:>4} NaNs ({nan_percent:>6.2f}%)")
    else:
        print(f"    ✓ No missing values in selected columns")

    print(f"\n[6] Preparing data for LCA...")

    available_cols = [col for col in question_list if col in work_df.columns]

    if not available_cols:
        raise ValueError("No columns available for LCA analysis")

    data_subset = work_df[available_cols].copy()

    try:
        X, categories_per_var, var_names, cat_mappings = prepare_data_for_lca(
            data_subset, available_cols
        )

        n_samples, n_vars = X.shape
        print(f"    ✓ Data prepared: {n_vars} categorical variables, {n_samples} observations")
        print(f"    ✓ Categories per variable: {categories_per_var}")

        total_categories = sum(categories_per_var)
        avg_categories = total_categories / n_vars
        print(f"    ✓ Total categories: {total_categories} (average: {avg_categories:.1f} per variable)")

    except Exception as e:
        print(f"    ✗ Error preparing data: {str(e)}")
        raise

    print(f"\n[7] Performing Latent Class Analysis (EM)...")

    results = {}
    prob_results = {}
    metrics = []
    models = {}

    for n_components in n_components_range:
        print(f"\n    k = {n_components} latent classes...")

        try:
            lca = LatentClassAnalysis(
                n_classes=n_components,
                max_iter=500,
                tol=1e-4,
                n_init=5,
                random_state=42,
                verbose=False
            )

            lca.fit(X, categories_per_var, var_names, cat_mappings)

            models[n_components] = lca

            clusters = lca.predict()
            probabilities = lca.predict_proba()

            results[n_components] = clusters
            prob_results[n_components] = probabilities

            avg_certainty = lca.get_average_certainty()
            class_entropy = lca.get_entropy()

            metrics.append({
                'k': n_components,
                'BIC': lca.bic_,
                'AIC': lca.aic_,
                'Log_Likelihood': lca.log_likelihood_,
                'Converged': lca.converged_,
                'Iterations': lca.n_iter_,
                'Avg_certainty': avg_certainty,
                'Entropy': class_entropy
            })

            print(f"      ✓ BIC: {lca.bic_:.2f}, AIC: {lca.aic_:.2f}")
            print(f"      ✓ Log-Likelihood: {lca.log_likelihood_:.2f}")
            print(f"      ✓ Average certainty: {avg_certainty:.3f}")
            print(f"      ✓ Entropy: {class_entropy:.3f}")

            unique, counts = np.unique(clusters, return_counts=True)
            print(f"      ✓ Class distribution:")
            for cluster, count in sorted(zip(unique, counts)):
                percentage = (count / len(clusters)) * 100
                print(f"          Class {cluster}: {count} ({percentage:.1f}%)")

            if n_components <= 4:
                try:
                    profiles = lca.get_class_profiles_human_readable()
                except:
                    pass

        except Exception as e:
            print(f"      ✗ Error for k={n_components}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[n_components] = None
            prob_results[n_components] = None

    print(f"\n[8] Creating output DataFrames...")

    output_cols = [id_column] + available_cols if id_column in work_df.columns else available_cols
    output_df = work_df[output_cols].copy()

    prob_df = pd.DataFrame()
    if id_column in work_df.columns:
        prob_df[id_column] = work_df[id_column].copy()

    for n_components in n_components_range:
        if results.get(n_components) is not None:
            col_name = f'lca_k{n_components}'
            output_df[col_name] = results[n_components]

            probabilities = prob_results[n_components]
            for class_idx in range(n_components):
                prob_col = f'lca_k{n_components}_prob_class{class_idx}'
                prob_df[prob_col] = probabilities[:, class_idx]

            print(f"    ✓ k={n_components}: {n_components} classes and {n_components} probability columns")

    print(f"\n" + "="*70)
    print("FINAL SUMMARY - OPTIMIZED LCA")
    print("="*70)

    print(f"\nMain DataFrame (output_df):")
    print(f"  • Dimensions: {output_df.shape[0]} rows × {output_df.shape[1]} columns")
    print(f"  • Categorical variables: {len(available_cols)}")
    print(f"  • Missing values preserved: Yes")

    print(f"\nProbability DataFrame (prob_df):")
    print(f"  • Dimensions: {prob_df.shape[0]} rows × {prob_df.shape[1]} columns")

    if metrics:
        metrics_df = pd.DataFrame(metrics)
        print(f"\nModel metrics:")
        print(metrics_df.to_string(index=False, float_format="%.2f"))

        valid_metrics = metrics_df[metrics_df['BIC'].notnull()]
        if not valid_metrics.empty:
            best_model_idx = valid_metrics['BIC'].idxmin()
            best_model = valid_metrics.loc[best_model_idx]
            print(f"\n  → Recommended model: k = {int(best_model['k'])} (lowest BIC)")
            print(f"     • BIC: {best_model['BIC']:.2f}")
            print(f"     • AIC: {best_model['AIC']:.2f}")
            print(f"     • Certainty: {best_model['Avg_certainty']:.3f}")
            print(f"     • Entropy: {best_model['Entropy']:.3f}")

            if int(best_model['k']) in models:
                print(f"     • Model stored in 'models' dictionary")

    metric_df = pd.DataFrame(metrics)
    column_order = ['k', 'BIC', 'AIC', 'Log_Likelihood', 'Converged',
                    'Iterations', 'Avg_certainty', 'Entropy']
    metric_df = metric_df[column_order]

    print(f"\n" + "="*70)
    print("PROCESS COMPLETED")
    print("="*70)

    return output_df, prob_df, metric_df


def plot_lca_results(prob_df: pd.DataFrame, n_classes: int,
                     save_path: str = None) -> None:
    """
    Generates visualizations for LCA results.
    """
    prob_cols = [f'lca_k{n_classes}_prob_class{i}' for i in range(n_classes)]

    if not all(col in prob_df.columns for col in prob_cols):
        print(f"  ⚠ Probability columns for k={n_classes} not found")
        return

    prob_data = prob_df[prob_cols].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i in range(n_classes):
        axes[0].hist(prob_data[:, i], bins=20, alpha=0.5,
                    label=f'Class {i}', density=True)

    axes[0].set_xlabel('Posterior probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of posterior probabilities')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    max_probs = prob_data.max(axis=1)
    axes[1].hist(max_probs, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(max_probs.mean(), color='red', linestyle='--',
                    label=f'Mean: {max_probs.mean():.3f}')
    axes[1].set_xlabel('Maximum probability')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Assignment certainty (average: {max_probs.mean():.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    sample_entropies = entropy(prob_data.T)
    axes[2].hist(sample_entropies, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[2].axvline(sample_entropies.mean(), color='red', linestyle='--',
                   label=f'Mean: {sample_entropies.mean():.3f}')
    axes[2].set_xlabel('Entropy')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Sample entropy (average: {sample_entropies.mean():.3f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")

    plt.show()


def compare_lca_models(prob_dfs: Dict[str, pd.DataFrame],
                       n_classes_list: List[int]) -> pd.DataFrame:
    """
    Compares different LCA models.

    Parameters
    ----------
    prob_dfs : Dict[str, pd.DataFrame]
        Dictionary with probability DataFrames from different methods
    n_classes_list : List[int]
        List of numbers of classes to compare

    Returns
    -------
    comparison_df : DataFrame with comparison metrics
    """
    comparison_data = []

    for method_name, prob_df in prob_dfs.items():
        for n_classes in n_classes_list:
            prob_cols = [f'lca_k{n_classes}_prob_class{i}'
                        for i in range(n_classes)]

            if all(col in prob_df.columns for col in prob_cols):
                prob_data = prob_df[prob_cols].values

                max_probs = prob_data.max(axis=1)
                avg_certainty = max_probs.mean()
                sample_entropies = entropy(prob_data.T)
                avg_entropy = sample_entropies.mean()

                purity = (max_probs > 0.8).mean()

                comparison_data.append({
                    'Method': method_name,
                    'k': n_classes,
                    'Avg_certainty': avg_certainty,
                    'Avg_entropy': avg_entropy,
                    'Purity(>0.8)': purity,
                    'Samples': len(prob_data)
                })

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def calculate_imc_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates BMI categories from weight and height.
    Modify according to specific needs.
    """
    df_copy = df.copy()

    if 'pea002k' in df.columns and 'pea003k' in df.columns:
        imc = df_copy['pea002k'] / ((df_copy['pea003k'] / 100) ** 2)

        bins = [0, 18.5, 25, 30, 35, 40, 100]
        labels = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']

        df_copy['IMC_pea002'] = pd.cut(imc, bins=bins, labels=labels, right=False)
        df_copy['IMC_pea003'] = pd.cut(imc, bins=bins, labels=labels, right=False)

    return df_copy


CODE_MAP = {
    0: "NO", 2: "SI", 3: "AV", 1: "AV",
    7: "NR", 77: "NR",
    8: "NA", 88: "NA",
    9: "NS", 99: "NS"
}


def apply_code_map_without_imputation(df, columns, code_map=CODE_MAP):
    """
    Applies code mapping to specified columns without imputing NaN values.
    """
    df_result = df.copy()

    for col in columns:
        if col in df_result.columns:
            df_result[col] = df_result[col].apply(
                lambda x: code_map.get(x, x) if pd.notnull(x) else x
            )

    return df_result