


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as cindex
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import until_functions
from TransductiveSurvivalRankerRejection import TransductiveSurvivalRankerRejection as TSURE
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import torchtuples as tt
from pycox.models import CoxPH
from scipy.stats import combine_pvalues
import warnings
warnings.filterwarnings("ignore")

# ------Model parameters------#

LAMBDA_W = 0.5 # Controls weight regularization (increase for larger feature sets)
LAMBDA_U = 0.5 # Controls transductive loss strength (increase for clearer separation)
p = 2
LR=1e-4
DROPOUT=0.0


# ------Data preparation------#

time='DSS.time'
event='DSS'

# Available cancer types
# ['BLCA','BRCA','CSEC','COAD','KIRC','KIRP','LIHC','LUAD','STAD','UCEC']

cancer='BRCA'

# Available modalities
# ['WSI','Genes','Reports','Clinical']
modality='Genes'

#Retrieve the dataset

dataset,time,event,censoring = until_functions.fetch_Dataset(cancer,modality,time,event)

#Number of bootstrap runss
n_samples = 10


# False for models you do not want to run
with_Cox = True
with_RFS = True
with_DeepSurv = True


results_df = pd.DataFrame(columns=["Train Rate", "Run", "C-Index", "P-Value"])
results_df_Cox = pd.DataFrame(columns=["Train Rate", "Run", "C-Index", "P-Value"])
results_df_RFS = pd.DataFrame(columns=["Train Rate", "Run", "C-Index", "P-Value"])
results_df_DS = pd.DataFrame(columns=["Train Rate", "Run", "C-Index", "P-Value"])

run_id = 0  # Track run index

# Proportions of training data to use
proportions=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for proportion in proportions:
    for i in tqdm(range(n_samples),desc=f"Processing {proportion}"):

        if proportion!=0:

            train_size = int(len(dataset) * (proportion))
            # Resample the dataset to create a training set with the specified proportion
            # Resample fo many times to ensure that the training set has at least one event and one non-event
            # Which is very likely in small train size sets
            for k in range(10000000): 
                train_data = resample(dataset, n_samples=train_size,
                        replace=True, stratify=dataset[event])
                test_data = dataset.drop(train_data.index)
                if train_data[event].sum() > 0 and test_data[event].sum() > 0:
                    break


            test_data = dataset.drop(train_data.index)

            T_train = np.array(train_data.loc[:, time])
            E_train = np.array(train_data.loc[:, event])
            X_train = np.array(train_data.drop([time, event], axis=1))
            
            # #Censoring
            E_train[T_train>censoring]=0
            T_train[T_train>censoring]=censoring
            
            
            # Test data
            E_test = np.array(test_data.loc[:, event])
            T_test = np.array(test_data.loc[:, time])
            X_test = np.array(test_data.drop([time, event], axis=1))
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train=scaler.transform(X_train)
            X_test=scaler.transform(X_test)
                
        
            
            # ------------------Run Transductive Learning--------------

            TSURE_model = TSURE(lambda_w=LAMBDA_W, lambda_u=LAMBDA_U, p=p, Tmax=2000, lr=LR)
            TSURE_model.fit(X_train, T_train, E_train, X_test,plot_loss=False)

            Z_test_TSURE = TSURE_model.decision_function(X_test)
            cindex_TSURE= cindex(T_test, Z_test_TSURE, E_test*(T_test < censoring))

            #Split into high/low risk groups based on the prediction score
            Results_df = pd.DataFrame(
                {'Prediction': Z_test_TSURE, 'Time': T_test, 'Event': E_test})
            low_group = Results_df[Results_df['Prediction'] > 0]
            high_group = Results_df[Results_df['Prediction'] <= 0]

            results = logrank_test(low_group['Time'], high_group['Time'], event_observed_A=low_group['Event'],
                                    event_observed_B=high_group['Event'])
            pvalue = results.p_value
            results_df.loc[run_id] = [proportion, i + 1, cindex_TSURE, pvalue]


            if with_Cox:
                time_df=pd.DataFrame(T_train,columns=[time])
                event_df=pd.DataFrame(E_train,columns=[event])
                x_df=pd.DataFrame(X_train, columns=dataset.drop([time,event],axis=1).columns)
                train_data= pd.concat([x_df, time_df, event_df], axis=1)
                
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(train_data, duration_col=time, event_col=event,step_size=0.1)

                # Calculate the concordance index on the validation set
                cindex_Cox=cindex(test_data[time], -cph.predict_partial_hazard(test_data), test_data[event]*(test_data[time] < censoring))

                # Select threshold
                Z_train = -cph.predict_partial_hazard(train_data)
                Z_test = -cph.predict_partial_hazard(test_data)
                median = np.median(Z_train)

                Results_df = pd.DataFrame(
                    {'Prediction': Z_test, 'Time': test_data[time], 'Event': test_data[event]})

                low_group_Cox = Results_df[Results_df['Prediction'] > median]
                high_group_Cox = Results_df[Results_df['Prediction'] <= median]

                results_Cox = logrank_test(low_group_Cox['Time'], high_group_Cox['Time'], event_observed_A=low_group_Cox['Event'],
                                            event_observed_B=high_group_Cox['Event'])
                pvalue_Cox = results_Cox.p_value

                results_df_Cox.loc[run_id] = [proportion, i + 1, cindex_Cox, pvalue_Cox]

            if with_RFS:

                y_train_struct = Surv.from_arrays(event=E_train.astype(bool), time=T_train)
                y_test_struct = Surv.from_arrays(event=E_test.astype(bool), time=T_test)
                rfs = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1)
                rfs.fit(X_train, y_train_struct)

                # Predict risk scores (negated for compatibility with Cox logic)
                Z_train = -rfs.predict(X_train)
                Z_test = -rfs.predict(X_test)

                # Evaluate C-index
                cindex_RFS = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))

                # Risk stratification
                median = np.median(Z_train)
                Results_df = pd.DataFrame({
                    'Prediction': Z_test,
                    'Time': test_data[time],
                    'Event': test_data[event]
                })

                low_group_RFS = Results_df[Results_df['Prediction'] <= median]
                high_group_RFS = Results_df[Results_df['Prediction'] > median]

                # Log-rank test
                results_RFS = logrank_test(
                    low_group_RFS['Time'], high_group_RFS['Time'],
                    event_observed_A=low_group_RFS['Event'],
                    event_observed_B=high_group_RFS['Event']
                )
                pvalue = results_RFS.p_value
                results_df_RFS.loc[run_id] = [proportion, i + 1, cindex_RFS, pvalue]

            if with_DeepSurv:
                in_features = X_train.shape[1]
                num_nodes = [32, 32]          # Two hidden layers with 32 units each
                out_features = 1              # Single output for CoxPH
                batch_norm = True
                dropout = 0.1
                output_bias = False

                net = tt.practical.MLPVanilla(
                    in_features, num_nodes, out_features,
                    batch_norm=batch_norm, dropout=dropout, output_bias=output_bias
                )

                # Wrap the model
                model = CoxPH(net, tt.optim.Adam)

                # Fit the model
                X_train_f = np.asarray(X_train, dtype=np.float32)
                get_target = lambda df: (df[time].values, df[event].values)
                y_train = get_target(train_data)
                y_train = (np.asarray(y_train[0], dtype=np.int64), np.asarray(y_train[1], dtype=np.int64))
                model.fit(X_train_f, y_train, batch_size=128, epochs=1000, verbose=False)
                X_test_f= np.asarray(X_test, dtype=np.float32)

                _ = model.compute_baseline_hazards()

                surv_train = model.predict_surv_df(X_train_f)
                surv_test = model.predict_surv_df(X_test_f)

                # Aggregate: sum negative survival probabilities over time (per patient)
                Z_train = surv_train.sum(axis=0).values
                Z_test = surv_test.sum(axis=0).values

                # Compute C-index using your censoring logic
                cindex_DeepSurv = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))

                # Median-based risk stratification
                median = np.median(Z_train)
                Results_df = pd.DataFrame({
                    'Prediction': Z_test,
                    'Time': test_data[time],
                    'Event': test_data[event]
                })

                low_group = Results_df[Results_df['Prediction'] <= median]
                high_group = Results_df[Results_df['Prediction'] > median]

                # Log-rank test
                results_DeepSurv = logrank_test(
                    low_group['Time'], high_group['Time'],
                    event_observed_A=low_group['Event'],
                    event_observed_B=high_group['Event']
                )
                pvalue = results_DeepSurv.p_value
                results_df_DS.loc[run_id] = [proportion, i + 1, cindex_DeepSurv, pvalue]

            run_id += 1 
    





labelled_counts = [int(p * len(dataset)) for p in proportions]  # Adjust if proportions are scaled differently

# -- Helper to extract values from result DataFrames --
def extract_metrics(results_df):
    grouped = results_df.groupby("Train Rate").agg({
        "C-Index": ["mean", "std"],
        "P-Value": ["median", "std"]
    })
    cindex_mean = grouped["C-Index"]["mean"].reindex(proportions)
    cindex_std = grouped["C-Index"]["std"].reindex(proportions)
    pvalue_mean = grouped["P-Value"]["median"].reindex(proportions)
    pvalue_std = grouped["P-Value"]["std"].reindex(proportions)
    pvalue_std.fillna(1, inplace=True)
    # Compute combined p-values using Fisher's method
    combined_pvalues = []
    for rate in proportions:
        pvals = results_df.loc[results_df["Train Rate"] == rate, "P-Value"]
        pvals = pvals.dropna()
        if len(pvals) > 0:
            _, combined = combine_pvalues(pvals, method='fisher')
        else:
            combined = np.nan
        combined_pvalues.append(combined)

    combined_pvalues = pd.Series(combined_pvalues, index=proportions, name="Combined P-Value")


    return cindex_mean, cindex_std, pvalue_mean, pvalue_std,combined_pvalues

# -- Extract TSURE, NoTSURE, Cox metrics --
cindex_mean_TSURE, cindex_std_TSURE, pvalue_mean_TSURE, pvalue_std_TSURE, combine_pvalues_TSURE = extract_metrics(results_df)
cindex_mean_Cox, cindex_std_Cox, pvalue_mean_Cox, pvalue_std_Cox, combine_pvalues_Cox = extract_metrics(results_df_Cox)
cindex_mean_RFS, cindex_std_RFS, pvalue_mean_RFS, pvalue_std_RFS, combine_pvalues_RFS = extract_metrics(results_df_RFS)
cindex_mean_DS, cindex_std_DS, pvalue_mean_DS, pvalue_std_DS, combine_pvalues_DS = extract_metrics(results_df_DS)



# Prepare data as a list of dictionaries (one per method)
metrics = [
    {   "Data": f"ç",
        "Method": "TSURE",
        "C-index Mean": cindex_mean_TSURE,
        "C-index Std": cindex_std_TSURE,
        "P-value Median": pvalue_mean_TSURE,
        "P-value Std": pvalue_std_TSURE,
        "Combined P-value": combine_pvalues_TSURE
    },
    {   "Data": f"{cancer}-{modality}",
        "Method": "Cox",
        "C-index Mean": cindex_mean_Cox,
        "C-index Std": cindex_std_Cox,
        "P-value Median": pvalue_mean_Cox,
        "P-value Std": pvalue_std_Cox,
        "Combined P-value": combine_pvalues_Cox
    },
    {   "Data": f"{cancer}-{modality}",
        "Method": "RFS",
        "C-index Mean": cindex_mean_RFS,
        "C-index Std": cindex_std_RFS,
        "P-value Median": pvalue_mean_RFS,
        "P-value Std": pvalue_std_RFS,
        "Combined P-value": combine_pvalues_RFS
    },
    {   "Data": f"{cancer}-{modality}",
        "Method": "DeepSurv",
        "C-index Mean": cindex_mean_DS,
        "C-index Std": cindex_std_DS,
        "P-value Median": pvalue_mean_DS,
        "P-value Std": pvalue_std_DS,
        "Combined P-value": combine_pvalues_DS
    }
]

# Create DataFrame
metrics_df = pd.DataFrame(metrics)

# Save to CSV
metrics_df.to_csv(f"Results/{cancer}_{modality}_survival_metrics_summary_4models_DataEfficency.csv", index=False)


# --- C-INDEX PLOT ---
x_labels = [str(x) for x in labelled_counts]  # Convert to categorical x-axis

plt.figure(figsize=(12, 5))


# Uncomment fill_between if you want to visualize the standard deviation

plt.plot(x_labels, cindex_mean_TSURE, label="T-SURE", color="#6089a1", linewidth=2)
# plt.fill_between(x_labels, cindex_mean_TSURE - cindex_std_TSURE, cindex_mean_TSURE + cindex_std_TSURE, alpha=0.2, color="#6089a1")

plt.plot(x_labels, cindex_mean_Cox, label="CoxPH", color="#6da56d", linewidth=2)
# plt.fill_between(x_labels, cindex_mean_Cox - cindex_std_Cox, cindex_mean_Cox + cindex_std_Cox, alpha=0.2, color="#6da56d")

plt.plot(x_labels, cindex_mean_RFS, label="RFS", color="#a36060", linewidth=2)
# plt.fill_between(x_labels, cindex_mean_RFS - cindex_std_RFS, cindex_mean_RFS + cindex_std_RFS, alpha=0.2, color="#a36060")

plt.plot(x_labels, cindex_mean_DS, label="DeepSurv", color="#f0a86b", linewidth=2)
# plt.fill_between(x_labels, cindex_mean_DS - cindex_std_DS, cindex_mean_DS + cindex_std_DS, alpha=0.2, color="#f0a86b")


plt.xlabel("Number of Labeled Samples", fontsize=12)
plt.ylabel("C-Index", fontsize=12)
plt.title(f"Impact of Labeled Sample Size on C-Index ({cancer}-{modality})", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.xticks(rotation=45)  # Tilt for readability
plt.tight_layout()
plt.show()

x=0
# --- P-VALUE PLOT ---

neglog_pvalue_mean_TSURE = -np.log10(pvalue_mean_TSURE)
neglog_pvalue_std_TSURE = pvalue_std_TSURE / (pvalue_mean_TSURE * np.log(10))  # approx std of -log10(p)

neglog_pvalue_mean_Cox = -np.log10(pvalue_mean_Cox)
neglog_pvalue_std_Cox = pvalue_std_Cox / (pvalue_mean_Cox * np.log(10))

neglog_pvalue_mean_RFS = -np.log10(pvalue_mean_RFS)
neglog_pvalue_std_RFS = pvalue_std_RFS / (pvalue_mean_RFS * np.log(10))

neglog_pvalue_mean_DS = -np.log10(pvalue_mean_DS)
neglog_pvalue_std_DS = pvalue_std_DS / (pvalue_mean_DS * np.log(10))

max_allowed_std = 2  
neglog_pvalue_std_TSURE = np.clip(neglog_pvalue_std_TSURE, 0, max_allowed_std)
neglog_pvalue_std_Cox = np.clip(neglog_pvalue_std_Cox, 0, max_allowed_std)
neglog_pvalue_std_RFS = np.clip(neglog_pvalue_std_RFS, 0, max_allowed_std)
neglog_pvalue_std_DS = np.clip(neglog_pvalue_std_DS, 0, max_allowed_std)

# Prepare x-axis labels based on the number of labelled samples
x_labels = [str(x) for x in labelled_counts]  # categorical x-axis

plt.figure(figsize=(12, 5))

# Uncomment fill_between if you want to visualize the standard deviation

plt.plot(x_labels, neglog_pvalue_mean_TSURE, label="T-SURE", color="#6089a1", linewidth=2, linestyle="dashed")
# plt.fill_between(x_labels, neglog_pvalue_mean_TSURE - neglog_pvalue_std_TSURE, 
#                  neglog_pvalue_mean_TSURE + neglog_pvalue_std_TSURE, alpha=0.2, color="#f0a86b")

plt.plot(x_labels, neglog_pvalue_mean_Cox, label="CoxPH", color="#6da56d", linewidth=2, linestyle="dashed")
# plt.fill_between(x_labels, neglog_pvalue_mean_Cox - neglog_pvalue_std_Cox, 
#                  neglog_pvalue_mean_Cox + neglog_pvalue_std_Cox, alpha=0.2, color="#6d9ea5")

plt.plot(x_labels, neglog_pvalue_mean_RFS, label="RFS", color="#a36060", linewidth=2, linestyle="dashed")
# plt.fill_between(x_labels, neglog_pvalue_mean_RFS - neglog_pvalue_std_RFS,
#                     neglog_pvalue_mean_RFS + neglog_pvalue_std_RFS, alpha=0.2, color="#a36060")

plt.plot(x_labels, neglog_pvalue_mean_DS, label="DeepSurv", color="#f0a86b", linewidth=2, linestyle="dashed")
# plt.fill_between(x_labels, neglog_pvalue_mean_DS - neglog_pvalue_std_DS,
#                     neglog_pvalue_mean_DS + neglog_pvalue_std_DS, alpha=0.2, color="#6089a1")

# Significance thresholds
plt.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1)
plt.text(0, -np.log10(0.05) , "p = 0.05", color="gray", fontsize=10)

# plt.axhline(-np.log10(0.0001), color="gray", linestyle="--", linewidth=1)
# plt.text(0, -np.log10(0.0001) , "p = 0.0001", color="gray", fontsize=10)

plt.xlabel("Number of Labeled Samples", fontsize=12)
plt.ylabel("-log(p-value)", fontsize=12)
plt.title(f"Impact of Labeled Sample Size on p-value ({cancer}-{modality})", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.xticks(rotation=45)
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()



x=0



