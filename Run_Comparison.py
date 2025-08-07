
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as cindex
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import until_functions
from TransductiveSurvivalRankerRejection import TransductiveSurvivalRankerRejection as TSURE
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
import torchtuples as tt
from pycox.models import MTLR
from pycox.models import CoxPH
from scipy.stats import combine_pvalues
import json


# This script runs Cross Validation to compare T-SURE with other models in terms of ranking only

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
modality='WSI'

#Retrieve the dataset
dataset,time,event,censoring = until_functions.fetch_Dataset(cancer,modality,time,event)

# Load cross validation splits
with open(f"./Splits/{modality}_DSS_{cancer}_cv_splits.json", "r") as f:
    split_info = json.load(f)
if len(split_info)>5:split_info=split_info[-5:] 


# df to save results for all models
summary_df = pd.DataFrame(columns=['Cancer Type', 'Model', 'Mean C-Index', 'Std C-Index', 'Combined P-Value'])

cancer_dataframes = {}

# Prepare result storage
model_results = {
    'TSURE': {'cindices': [], 'pvalues': []},
    'Cox': {'cindices': [], 'pvalues': []},
    'GBM': {'cindices': [], 'pvalues': []},
    'RFS': {'cindices': [], 'pvalues': []},
    'SVM': {'cindices': [], 'pvalues': []},
    'MTLR': {'cindices': [], 'pvalues': []},
    'DeepSurv': {'cindices': [], 'pvalues': []},
}



for fold_data in tqdm(split_info):
    fold = fold_data["fold"]
    train_ids = fold_data["train_ids"]
    test_ids = fold_data["test_ids"]

    train_data = dataset.loc[train_ids]
    test_data = dataset.loc[test_ids]

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

    TSURE_model = TSURE(lambda_w=LAMBDA_W, lambda_u=LAMBDA_U, p=p, Tmax=2000, lr=LR,dropout=DROPOUT)
    TSURE_model.fit(X_train, T_train, E_train, X_test)

    Z_test_TSURE = TSURE_model.decision_function(X_test)
    cindex_TSURE= cindex(T_test, Z_test_TSURE, E_test*(T_test < censoring))
    model_results['TSURE']['cindices'].append(cindex_TSURE)

    #Split into high/low risk groups based on the prediction score
    #Threshold is always set to 0

    Results_df = pd.DataFrame(
        {'Prediction': Z_test_TSURE, 'Time': T_test, 'Event': E_test})
    low_group = Results_df[Results_df['Prediction'] <= 0]
    high_group = Results_df[Results_df['Prediction'] > 0]

    #logrank test to calculate significance
    results = logrank_test(low_group['Time'], high_group['Time'], event_observed_A=low_group['Event'],
                            event_observed_B=high_group['Event'])
    model_results['TSURE']['pvalues'].append(results.p_value)



#-------------------------------------------------------#
#----------------------------COX------------------------#
#-------------------------------------------------------#

    #Convert dataset to separate dataframes as required by CoxPH

    time_df=pd.DataFrame(T_train,columns=[time])
    event_df=pd.DataFrame(E_train,columns=[event])
    x_df=pd.DataFrame(X_train, columns=dataset.drop([time,event],axis=1).columns)
    train_data= pd.concat([x_df, time_df, event_df], axis=1)
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_data, duration_col=time, event_col=event, step_size=0.1)
    
    # Compute C-index
    cindex_Cox=cindex(test_data[time], -cph.predict_partial_hazard(test_data), test_data[event]*(test_data[time] < censoring))
    model_results['Cox']['cindices'].append(cindex_Cox)

    # Risk stratification based on median of predicted scores
    Z_train = -cph.predict_partial_hazard(train_data)
    Z_test = -cph.predict_partial_hazard(test_data)
    median = np.median(Z_train)
    
    
    Results_df = pd.DataFrame(
        {'Prediction': Z_test, 'Time': test_data[time], 'Event': test_data[event]})

    low_group_Cox = Results_df[Results_df['Prediction'] <= median]
    high_group_Cox = Results_df[Results_df['Prediction'] > median]

    results_Cox = logrank_test(low_group_Cox['Time'], high_group_Cox['Time'], event_observed_A=low_group_Cox['Event'],
                                event_observed_B=high_group_Cox['Event'])
    model_results['Cox']['pvalues'].append(results_Cox.p_value)

#-------------------------------------------------------#
#----------------------------GBM------------------------#
#-------------------------------------------------------#

    # Convert survival data to structured array
    y_train_struct = Surv.from_arrays(event=E_train.astype(bool), time=T_train)
    y_test_struct = Surv.from_arrays(event=E_test.astype(bool), time=T_test)

    # Fit GBM RFS
    gbm = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, max_depth=3)
    gbm.fit(X_train, y_train_struct)

    # Predict risk scores (negative for higher risk)
    Z_train = -gbm.predict(X_train)
    Z_test = -gbm.predict(X_test)

    # Compute C-index 
    cindex_GBM = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))
    model_results['GBM']['cindices'].append(cindex_GBM)

    # Risk stratification based on median of predicted scores
    median = np.median(Z_train)
    Results_df = pd.DataFrame({
        'Prediction': Z_test,
        'Time': test_data[time],
        'Event': test_data[event]
    })

    low_group_GBM = Results_df[Results_df['Prediction'] <= median]
    high_group_GBM = Results_df[Results_df['Prediction'] > median]

    # Log-rank test between risk groups
    results_GBM = logrank_test(
        low_group_GBM['Time'], high_group_GBM['Time'],
        event_observed_A=low_group_GBM['Event'],
        event_observed_B=high_group_GBM['Event']
    )
    model_results['GBM']['pvalues'].append(results_GBM.p_value)

#-------------------------------------------------------#
#----------------------------RFS------------------------#
#-------------------------------------------------------#

    rfs = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1)
    rfs.fit(X_train, y_train_struct)

    # Predict risk scores (negated for compatibility with Cox logic)
    Z_train = -rfs.predict(X_train)
    Z_test = -rfs.predict(X_test)

    # Compute C-index
    cindex_RFS = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))
    model_results['RFS']['cindices'].append(cindex_RFS)

    # Risk stratification based on median of predicted scores
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
    model_results['RFS']['pvalues'].append(results_RFS.p_value)


#-------------------------------------------------------#
#----------------------------SVM------------------------#
#-------------------------------------------------------#
    svm = FastSurvivalSVM(max_iter=1000, tol=1e-5, rank_ratio=1.0, alpha=1.0)
    svm.fit(X_train, y_train_struct)

    # Predict risk scores (negated for compatibility)
    Z_train = -svm.predict(X_train)
    Z_test = -svm.predict(X_test)

    # Compute C-index
    cindex_SVM = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))
    model_results['SVM']['cindices'].append(cindex_SVM)

    # Risk stratification based on median of predicted scores
    median = np.median(Z_train)
    Results_df = pd.DataFrame({
        'Prediction': Z_test,
        'Time': test_data[time],
        'Event': test_data[event]
    })

    low_group_SVM = Results_df[Results_df['Prediction'] <= median]
    high_group_SVM = Results_df[Results_df['Prediction'] > median]

    # Log-rank test
    results_SVM = logrank_test(
        low_group_SVM['Time'], high_group_SVM['Time'],
        event_observed_A=low_group_SVM['Event'],
        event_observed_B=high_group_SVM['Event']
    )
    model_results['SVM']['pvalues'].append(results_SVM.p_value)

#-------------------------------------------------------#
#----------------------------MTLR------------------------#
#-------------------------------------------------------#

    # Discretize time into intervals

    num_durations = 20
    labtrans = MTLR.label_transform(num_durations)
    get_target = lambda df: (df[time].values, df[event].values)
    y_train = labtrans.fit_transform(*get_target(train_data))

    train = (X_train, y_train)

    durations_test, events_test = get_target(test_data)

    in_features = X_train.shape[1]
    num_nodes = [32, 32]
    out_features = labtrans.out_features
    batch_norm = False
    dropout = 0.1

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)
    batch_size = 256
    model.optimizer.set_lr(0.01)
    epochs = 1000
    callbacks = [tt.callbacks.EarlyStopping()]

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train = (np.asarray(y_train[0], dtype=np.int64), np.asarray(y_train[1], dtype=np.int64))
    log = model.fit(X_train_f, y_train, batch_size, epochs, callbacks,verbose=False)


    X_test_f= np.asarray(X_test, dtype=np.float32)
    surv_train = model.predict_surv_df(X_train_f)
    surv_test = model.predict_surv_df(X_test_f)

    # Aggregate: sum negative survival probabilities over time (per patient)
    Z_train = surv_train.sum(axis=0).values
    Z_test = surv_test.sum(axis=0).values

    # Compute C-index
    cindex_MTLR = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))
    model_results['MTLR']['cindices'].append(cindex_MTLR)

    # Risk stratification based on median of predicted scores
    median = np.median(Z_train)
    Results_df = pd.DataFrame({'Prediction': Z_test, 'Time': test_data[time], 'Event': test_data[event]})
    low_group = Results_df[Results_df['Prediction'] <= median]
    high_group = Results_df[Results_df['Prediction'] > median]

    # Log-rank test
    results_MTLR = logrank_test(
        low_group['Time'], high_group['Time'],
        event_observed_A=low_group['Event'], event_observed_B=high_group['Event']
    )
    model_results['MTLR']['pvalues'].append(results_MTLR.p_value)

    #-------------------------------------------------------#
    #-------------------------DeepSurv----------------------#
    #-------------------------------------------------------#        

    # Format targets


    in_features = X_train.shape[1]
    num_nodes = [32, 32]          # Two hidden layers with 32 units each
    out_features = 1              # Single output for CoxPH
    batch_norm = True
    dropout = 0.0
    output_bias = False

    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features,
        batch_norm=batch_norm, dropout=dropout, output_bias=output_bias
    )



    get_target = lambda df: (df[time].values, df[event].values)
    y_train = get_target(train_data)
    y_train = (np.asarray(y_train[0], dtype=np.float32), np.asarray(y_train[1], dtype=np.int64))
    # Wrap the model
    model = CoxPH(net, tt.optim.Adam)

    # Fit the model
    model.fit(X_train_f, y_train, batch_size=128, epochs=1000, verbose=False)


    _ = model.compute_baseline_hazards()

    surv_train = model.predict_surv_df(X_train_f)
    surv_test = model.predict_surv_df(X_test_f)

    # Aggregate: sum negative survival probabilities over time (per patient)
    Z_train = surv_train.sum(axis=0).values
    Z_test = surv_test.sum(axis=0).values

    # Compute C-index 
    cindex_DeepSurv = cindex(test_data[time], Z_test, test_data[event] * (test_data[time] < censoring))
    model_results['DeepSurv']['cindices'].append(cindex_DeepSurv)

    # Risk stratification based on median of predicted scores
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
    model_results['DeepSurv']['pvalues'].append(results_DeepSurv.p_value)



# After cross-validation loop is completed, use this block to aggregate results
summary_df = pd.DataFrame()

for model_name, data in model_results.items():
    cindices = data['cindices']
    pvalues = [x for x in data['pvalues'] if not np.isnan(x)]

    mean_cindex = np.mean(cindices)
    std_cindex = np.std(cindices)
    combined_p = combine_pvalues(pvalues, method='fisher')[1] if pvalues else np.nan

    new_row = {
        'Cancer Type': cancer,
        'Model': model_name,
        'Mean C-Index': mean_cindex,
        'Std C-Index': std_cindex,
        'Combined P-Value': combined_p,
        'cindices': [cindices],
        'pvalues': [data['pvalues']]
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)

# Save results to a file
summary_df.to_csv(f'./Results/Summary_{modality}_{event}_{cancer}.csv', index=False)
