
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as cindex
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import until_functions
from TransductiveSurvivalRankerRejection import TransductiveSurvivalRankerRejection as TSURE
from lifelines.statistics import logrank_test
from scipy.stats import combine_pvalues

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

train_size = int(len(dataset) * 0.70)
Bootstrap_p_Values = []
Bootstrap_cindex = []

Rej_thresholds=6 # How many cutoffs to use for rejection
risk_matrix = [] # Store risk scores for heatmap generation
risk_series_list=[]

# Track rejection results
Results_TSURE = pd.DataFrame(columns=[str(x) for x in range(Rej_thresholds)]+['#Samples'])
Results_Random = pd.DataFrame(columns=[str(x) for x in range(Rej_thresholds)]+['#Samples'])

# How many bootstrap runs to use
# For statistical significance, +500 is recommended
n_samples = 10 


for i in tqdm(range(n_samples)):

    train_data = resample(dataset, n_samples=train_size,
                replace=True, stratify=dataset[event])

    test_data = dataset.drop(train_data.index)

    T_train = np.array(train_data.loc[:, time])
    E_train = np.array(train_data.loc[:, event])
    X_train = np.array(train_data.drop([time, event], axis=1))
    
    # Censoring
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

    TSURE_model = TSURE(lambda_w=LAMBDA_W, lambda_u=LAMBDA_U, p=p, lr=LR, Tmax=2000, dropout=DROPOUT)
    TSURE_model.fit(X_train, T_train, E_train,X_test, plot_loss=False)

    Z_test_TSURE = TSURE_model.decision_function(X_test)
    og_cindex= cindex(T_test, Z_test_TSURE, E_test*(T_test < censoring))
    cindex_rej_TSURE=og_cindex
    Bootstrap_cindex.append(cindex_rej_TSURE)

    #Split into high/low risk groups based on the prediction score
    #Threshold is always set to 0

    Results_df = pd.DataFrame(
        {'Prediction': Z_test_TSURE, 'Time': T_test, 'Event': E_test})
    low_group = Results_df[Results_df['Prediction'] > 0]
    high_group = Results_df[Results_df['Prediction'] <= 0]


    #logrank test to calculate significance
    results = logrank_test(low_group['Time'], high_group['Time'], event_observed_A=low_group['Event'],
                               event_observed_B=high_group['Event'])
    pvalue = results.p_value
    Bootstrap_p_Values.append(pvalue)

    # Keep track of risk scores
    risk_series = pd.Series(Z_test_TSURE, index=test_data.index, name=f"run_{i}")
    risk_series_list.append(risk_series)

#----------------------------REJECTION METHODOLOGY------------------------#

    Rej_score_TSURE = np.abs(Z_test_TSURE) #As specified in the paper

    results_TSURE = []
    results_TSURE.append([cindex_rej_TSURE,len(Z_test_TSURE)])


    results_Random = []
    results_Random.append([cindex_rej_TSURE,len(Z_test_TSURE)])
    random_rejected_indices = set() # used to keep track of indices that have been randomly rejected

    for i, t in enumerate(range(1, Rej_thresholds)): 
    #for i, t in enumerate(range(Rej_thresholds-1,0,-1)):

        rej_thresh_TSURE=np.quantile(Rej_score_TSURE, 0.10+(t/10))

        #-------------TSURE-------------#
        try:
            mask_TSURE = Rej_score_TSURE>rej_thresh_TSURE

            masked_T = T_test[mask_TSURE]
            masked_E = E_test[mask_TSURE]
            masked_Z = Z_test_TSURE[mask_TSURE]
            cindex_rej_TSURE=cindex(masked_T,masked_Z , masked_E )
            results_TSURE.append([cindex_rej_TSURE,len(masked_T)])
            accepted_indices_TSURE = set(test_data.index[mask_TSURE])
               
        except:
            print (f"no permissible pairs: {np.sum(masked_E)}, in quantile {t} and threshold: {rej_thresh_TSURE}")
            results_TSURE.append([cindex_rej_TSURE,len(masked_T)])


                        #------------RANDOM REJECTION------------#
        try:                

            all_indices = np.arange(len(Rej_score_TSURE))
            available_indices = np.setdiff1d(all_indices, list(random_rejected_indices))

            # Calculate the number of samples to reject (10% of the remaining samples)
            num_samples_to_reject = int(0.1 * len(available_indices))

            newly_rejected_indices = np.random.choice(available_indices, size=num_samples_to_reject, replace=False)

            # Update the rejected indices set
            random_rejected_indices.update(newly_rejected_indices)

            # Create a mask for the non-rejected samples
            mask_TSURE = np.ones(len(Rej_score_TSURE), dtype=bool)
            mask_TSURE[list(random_rejected_indices)] = False  # Set already rejected indices to False

            masked_T = T_test[mask_TSURE]
            masked_E = E_test[mask_TSURE]
            masked_Z = Z_test_TSURE[mask_TSURE]

            # Calculate the c-index after rejection
            cindex_rej_Random = cindex(masked_T, masked_Z, masked_E)
            results_Random.append([cindex_rej_Random, len(masked_T)])
        except:
            print (f"Random - no permissible pairs: {np.sum(masked_E)}, in quantile {t}")
            results_Random.append([cindex_rej_Random, len(masked_T)])



    # Combine bootstrap results

    results_TSURE.append(len(Z_test_TSURE))
    Results_TSURE = pd.concat([Results_TSURE, pd.Series(results_TSURE, index=Results_TSURE.columns).to_frame().T], ignore_index=True)  
    

    results_Random.append(len(Z_test_TSURE))
    Results_Random = pd.concat([Results_Random, pd.Series(results_Random, index=Results_Random.columns).to_frame().T], ignore_index=True)  



# ---------Print all results---------- #

mean_score_TSURE = np.mean(Bootstrap_cindex)
std_score_TSURE = np.std(Bootstrap_cindex)
_, combined_p = combine_pvalues(Bootstrap_p_Values, method='fisher')

print("\nTSURE Mean ,SD,2p50:\n")
print("Mean c-index: %.2f" % mean_score_TSURE)
print("Standardd deviation of c-index: %.2f" % std_score_TSURE)
print(f"Combined p-Value: {combined_p}")


# ---------Rejection plot---------- #

dfs = [Results_TSURE,Results_Random] 
labels = ['TSURE Score','Random Rejection']   

title = f'TSURE-Based Rejection ({cancer} / {modality})'


until_functions.Rejection_Plot_Pvalue(dfs, labels, title)


# ---------Plot KM Curve ---------- #
title = f'KM Curve for ({cancer} / {modality})\n'
until_functions.plot_KM(Z_test_TSURE, T_test, E_test, censoring, title=title)

# ---------Plot Gene Heatmaps---------- #
if modality=='Genes':

    risk_matrix = pd.concat(risk_series_list, axis=1)
    avg_risk_score = risk_matrix.mean(axis=1, skipna=True)

    # Load the top genes list for the specified cancer type
    df = pd.read_csv("./Datasets/TCGA_GeneExp/top_genes_list.csv")
    df["parsed_genes"] = df["gene_list"].apply(lambda x: [g.strip() for g in x.split(",") if g.strip()])
    feature_list = df[df["Cancer"] == cancer]["parsed_genes"].values[0]

    until_functions.plot_feature_heatmap_grouped_clusters(dataset, avg_risk_score, feature_list,cancer=cancer, modality=modality)




x=0
