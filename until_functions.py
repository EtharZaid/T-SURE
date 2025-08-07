
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from scipy.stats import ttest_ind
from matplotlib.lines import Line2D
from lifelines.statistics import logrank_test
import seaborn as sns
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def fetch_Dataset(cancer,modality,time='DSS.time',event='DSS'):
    """
    Fetches the dataset for a given cancer type and modality.
    Parameters:
    - cancer: Cancer type (e.g., 'BRCA', 'LUAD').
    - modality: Modality type (e.g., 'WSI', 'Genes', 'Reports', 'Clinical').
    - time: Column name for survival time (default is 'DSS.time').
    - event: Column name for event indicator (default is 'DSS').
    """
    
    #----------Import Survival Data------------#


    df_survival = pd.read_excel("Datasets/TCGA-CDR-SupplementalTableS1.xlsx") 

    df_survival=df_survival.loc[:,['bcr_patient_barcode',time,event,'type']]
    df_survival.set_index('bcr_patient_barcode',inplace=True)
    df_survival.index.set_names('SAMPLE_ID',inplace=True)


    if modality=='WSI':
        data = pd.read_csv("./Datasets/All_TITAN_embeddings.csv")
        data.set_index('SAMPLE_ID',inplace=True)
        


    elif modality=='Genes':
        path=rf'./Datasets/TCGA_GeneExp/TCGA_{cancer}_.xlsx'
        data = pd.read_excel(path)
        data['SAMPLE_ID']=[p[:12] for p in data['SAMPLE_ID']]
        data.set_index('SAMPLE_ID',inplace=True)

    elif modality=='Reports':
        data = pd.read_excel('./Datasets/Reports_embeddings_All_TCGA.xlsx')
        data['SAMPLE_ID']=[p[:12] for p in data['SAMPLE_ID']]
        data.set_index('SAMPLE_ID',inplace=True)


    elif modality=='Clinical':
        data = pd.read_csv("./Datasets/Clinical_data_All_TCGA.tsv", sep='\t')

        # Columns that are indicative of survival and needs to be dropped
        drop_keywords = [
            "Survival", "Progress", "Disease Free", "Status", 
            "Last Communication", "Last Alive", "Months", "Calculated", "Time"
        ]
        #Remaining columns
        keep_columns = [
            'Diagnosis Age', 'Sex', 'Race Category',
            'Neoplasm Disease Stage',
            'Neoplasm Histologic Grade',
            'American Joint Committee on Cancer Tumor Stage Code',
            'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
            'American Joint Committee on Cancer Metastasis Stage Code',
            'Neoadjuvant Therapy Type Administered Prior To Resection Text',
            'Radiation Therapy',
            'Fraction Genome Altered', 'Mutation Count', 'TMB (nonsynonymous)',
            'Buffa Hypoxia Score', 'Ragnum Hypoxia Score', 'Winter Hypoxia Score',
            'MSI MANTIS Score', 'MSIsensor Score',
            'Tumor Disease Anatomic Site'
        ]
        df_cancer = data[data['TCGA PanCanAtlas Cancer Type Acronym'] == cancer]
        patient_ids = df_cancer['Patient ID']
        #-------Encooding and cleaning data-------#

        cols_to_drop = [col for col in df_cancer.columns if any(k in col for k in drop_keywords)]
        df_clean = df_cancer.drop(columns=cols_to_drop, errors='ignore')

        # Step 3: Keep only relevant predictors (if present in dataframe)
        available = [col for col in keep_columns if col in df_clean.columns]
        df_model = df_clean[available]

        # Step 4: One-hot encode categorical variables
        categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        df_encoded = pd.get_dummies(df_model, columns=categorical_cols, dummy_na=False)

        # -----------Drop columns with very low variance (e.g. < 0.01)--------------
        selector = VarianceThreshold(threshold=0.05)
        X_var_filtered = selector.fit_transform(df_encoded)

        selected_columns = df_encoded.columns[selector.get_support()]
        data = pd.DataFrame(X_var_filtered, columns=selected_columns)


        data['Patient ID'] = patient_ids.values
        data.set_index('Patient ID',inplace=True)
        data.index.set_names('SAMPLE_ID',inplace=True)

        
        if cancer=='LUAD':
            data.drop(['Neoadjuvant Therapy Type Administered Prior To Resection Text_No',
                        'Tumor Disease Anatomic Site_Lung'],axis=1,inplace=True)



    cancer_clinical = df_survival[df_survival['type'] == cancer]
    dataset = cancer_clinical.join(data, on='SAMPLE_ID')
    dataset=dataset.drop(['type'],axis=1)
    dataset.dropna(inplace=True)

    censoring = 3652 # 10 years in days

    return dataset,time,event,censoring

def Rejection_Plot_Pvalue(dfs, labels, title="Rejection Curve"):
    """
    Plots the rejection curve with significance annotations.
    Parameters:
    - dfs: List of DataFrames containing c-index and sample counts for each threshold.
    - labels: List of labels for each DataFrame.
    - title: Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    significance_used = set()

    for i, df in enumerate(dfs):
        threshold_columns = [col for col in df.columns if col != '#Samples']
        cindex_df = df[threshold_columns].applymap(lambda x: x[0])
        samples_df = df[threshold_columns].applymap(lambda x: x[1])
        total_samples = df['#Samples']
        percentage_samples_df = (samples_df.div(total_samples, axis=0)) * 100

        mean_cindex = cindex_df.mean(axis=0)
        std_cindex = cindex_df.std(axis=0)
        thresholds_numeric = [float(col) for col in threshold_columns]
        sorted_indices = np.argsort(thresholds_numeric)
        sorted_thresholds = np.array(thresholds_numeric)[sorted_indices]
        mean_cindex = mean_cindex.values[sorted_indices]
        std_cindex = std_cindex.values[sorted_indices]
        baseline_cindex = cindex_df.iloc[:, 0]

        if i == 0:  # T-SURE
            ax.errorbar(
                sorted_thresholds,
                mean_cindex,
                yerr=std_cindex,
                fmt='o-',
                capsize=5,
                capthick=1,
                label=labels[i],
                color='red',
                lw=2,
                markersize=5
            )

            for j, threshold in enumerate(sorted_thresholds):
                test_column = cindex_df.iloc[:, sorted_indices[j]]
                _, p_value = ttest_ind(baseline_cindex, test_column, equal_var=False)

                if p_value <= 0.001:
                    sig = '***'
                elif p_value <= 0.01:
                    sig = '**'
                elif p_value <= 0.05:
                    sig = '*'
                else:
                    sig = None

                if sig:
                    significance_used.add(sig)
                    ax.annotate(
                        sig,
                        (threshold, mean_cindex[j]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=12,
                        color='black'
                    )

        elif i == 1:  # Random
            ax.plot(
                sorted_thresholds,
                mean_cindex,
                linestyle='--',
                color='gray',
                linewidth=2,
                label=labels[i]
            )

    # Format axes
    ax.set_xlabel('Percentage of Rejected Samples')
    ax.set_ylabel('C-Index')
    ax.set_xticks(sorted_thresholds)
    ax.set_xticklabels([str(int(t * 10)) + '%' for t in sorted_thresholds])
    ax.grid(True)
    ax.set_title(title)

    # Significance legend
    significance_handles = []
    if '*' in significance_used:
        significance_handles.append(Line2D([0], [0], color='blue', linestyle='None', label='*  (0.01 < p ≤ 0.05)'))
    if '**' in significance_used:
        significance_handles.append(Line2D([0], [0], color='green', linestyle='None', label='** (0.001 < p ≤ 0.01)'))
    if '***' in significance_used:
        significance_handles.append(Line2D([0], [0], color='red', linestyle='None', label='*** (p ≤ 0.001)'))

    if significance_handles:
        ax.legend(handles=significance_handles, title="P-value Significance", loc='upper left')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


# Function to plot the distribution of prediction scores
def plot_Predictions_Dist( Z_test_TSURE, T_test, E_test, threshold, ax=None):
    """
    Plots the distribution of prediction scores for low and high risk groups.
    Parameters:
    - Z_test_TSURE: Array of prediction scores from the model.
    - T_test: Array of survival times for the test set.
    - E_test: Array of event indicators (1 if event occurred, 0 if censored).
    - threshold: Threshold value to split low and high risk groups.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))  # Create a new figure only if no ax is provided

    df = pd.DataFrame({'Prediction': Z_test_TSURE, 'Time': T_test, 'Event': E_test})
    low_group = df[df['Prediction'] > threshold]
    high_group = df[df['Prediction'] <= threshold]

    # Perform log-rank test
    results = logrank_test(low_group['Time'], high_group['Time'],
                           event_observed_A=low_group['Event'], event_observed_B=high_group['Event'])
    pvalue = results.p_value

    # Plot histograms
    ax.hist(low_group['Prediction'], bins=20, color='#4890c1', edgecolor='#aabbcc', linewidth=0.5, alpha=0.7, label='Low Risk')
    ax.hist(high_group['Prediction'], bins=20, color='#ecac7c', edgecolor='#fb9942', linewidth=0.5, alpha=0.7, label='High Risk')

    # Format p-value
    base, exponent = f"{pvalue:.1e}".split("e")
    p_text = rf"$p = {base} \times 10^{{{int(exponent)}}}$"

    # Set title
    ax.set_title(f"Prediction Scores Distribution - {p_text}", fontsize=12)

    ax.set_xlabel('Prediction Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.tick_params(axis='both', labelsize=12) 

    return ax  # Return only the axis (not a new figure)

def plot_KM(Z_test_TSURE, T_test, E_test, censoring, title=None):
    """
        Plots Kaplan-Meier survival curves for low and high risk groups based on predictions.
        Parameters:
        - Z_test_TSURE: Array of prediction scores from the model.
        - T_test: Array of survival times for the test set.
        - E_test: Array of event indicators (1 if event occurred, 0 if censored).
        - censoring: Censoring time point.
    """
    # Create a DataFrame for predictions
    Results_df = pd.DataFrame({'Prediction': Z_test_TSURE, 'Time': T_test, 'Event': E_test})

    # Split into risk groups
    low_group = Results_df[Results_df['Prediction'] > 0]
    high_group = Results_df[Results_df['Prediction'] <= 0]


    times_low = low_group['Time']
    events_low = low_group['Event']

    times_high = high_group['Time']
    events_high = high_group['Event']

    events_high.loc[times_high > censoring] = 0
    times_high.loc[times_high > censoring] = censoring
    events_low.loc[times_low > censoring] = 0
    times_low.loc[times_low > censoring] = censoring
    # Fit KM models
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    # Plot KM curves
    plt.figure(figsize=(6, 4))
    kmf_low.fit(times_low, event_observed=events_low, label='Low Risk')
    kmf_low.plot(ci_show=True, color='#98d98b', lw=2)

    kmf_high.fit(times_high, event_observed=events_high, label='High Risk')
    kmf_high.plot(ci_show=True, color='#D98B8B', lw=2)

    #logrank test to calculate significance
    results = logrank_test(low_group['Time'], high_group['Time'], event_observed_A=low_group['Event'],
                               event_observed_B=high_group['Event'])
    pvalue = results.p_value

    # Format p-value in scientific notation
    base, exponent = f"{pvalue:.1e}".split("e")
    p_text = rf"$p = {base} \times 10^{{{int(exponent)}}}$"

    # Set title
    if title is None:
        title = rf"{p_text}"
    else:
        title=title+rf"{p_text}"

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.tight_layout()
    plt.show()

# Plots a heatmap for the top unfavorable and favorable genes

def plot_feature_heatmap_grouped_clusters(
    feature_df,
    avg_risk_scores,
    feature_list,
    n_extreme=10,
    modality="Gene",
    cancer="BRCA"
):
    """
    Parameters:
    - feature_df: DataFrame (samples x features)
    - avg_risk_scores: Series with sample risk scores
    - feature_list: List of 2n features (first n unfavorable, last n favorable)
    - n_extreme: Number of high/low/rejected samples
    """
    assert len(feature_list) % 2 == 0, "Feature list must have even number (split into 2 groups)"
    n_genes = len(feature_list) // 2
    unfavorable_genes = [g for g in feature_list[:n_genes] if g in feature_df.columns]
    favorable_genes = [g for g in feature_list[n_genes:] if g in feature_df.columns]

    # Align sample indices
    common_samples = feature_df.index.intersection(avg_risk_scores.index)
    risk_scores = avg_risk_scores.loc[common_samples]
    feature_df = feature_df.loc[common_samples]

    # Sample groups
    low_ids = risk_scores.nsmallest(n_extreme).index
    high_ids = risk_scores.nlargest(n_extreme).index
    center_ids = risk_scores.abs().sort_values().head(n_extreme).index

    selected_ids = list(high_ids) + list(center_ids) + list(low_ids)
    risk_scores = risk_scores.loc[selected_ids]
    feature_df = -1*feature_df.loc[selected_ids]

    # Scale data
    all_genes = unfavorable_genes + favorable_genes
    data = feature_df.loc[selected_ids, all_genes]
    data_scaled = pd.DataFrame(
        StandardScaler().fit_transform(data),
        index=data.index,
        columns=data.columns
    ).T  # rows = genes, cols = samples

    # Cluster unfavorable/favorable separately
    from scipy.cluster.hierarchy import linkage, leaves_list

    def cluster_rows(subset_df):
        linkage_matrix = linkage(subset_df, method="average", metric="euclidean")
        return leaves_list(linkage_matrix)

    unf_rows = data_scaled.loc[unfavorable_genes]
    fav_rows = data_scaled.loc[favorable_genes]

    unf_order = unf_rows.index[cluster_rows(unf_rows)]
    fav_order = fav_rows.index[cluster_rows(fav_rows)]
    final_order = list(unf_order) + list(fav_order)

    # Color map: high = red, center = orange, low = green
    def label_color(score):
        if score >= risk_scores.loc[high_ids].min():
            return "indianred"
        elif score <= risk_scores.loc[low_ids].max():
            return "green"
        else:
            return "orange"

    risk_colors = risk_scores.map(label_color)

    # Plot heatmap
    sns.clustermap(
        data_scaled.loc[final_order],
        col_cluster=False,
        row_cluster=False,
        col_colors=risk_colors,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=True,
        figsize=(12, 6),
        vmin=-2,
        vmax=2
    )
    plt.suptitle(f"({modality}/{cancer}) Grouped Heatmap\nHigh | Rejected | Low", fontsize=14)
    plt.show()
