#%%
import pandas as pd
import numpy as np
from DELAY import DELAY_Classifier
import scanpy as sc

#%%
adata = sc.read_csv('example-data/NormalizedData.csv').T
t = pd.read_csv('example-data/PseudoTime.csv', index_col = 0)
tf = pd.read_csv('example-data/TranscriptionFactors.csv', header = None)
tf = tf.map(lambda x: x.capitalize())
adata.var['DELAY_TF'] = np.isin(adata.var_names, tf)
adata.var['DELAY_Target'] = True
adata.obs.loc[t.index, 't'] = t.values

y = pd.read_csv('example-data/refNetwork.csv', parse_dates = False)
y.columns = ['TF', 'Target']
y = y.map(lambda x: x.capitalize())
y = y.loc[np.isin(y.Target, adata.var_names)]

#%%
DELAY = DELAY_Classifier()
DELAY.compile_batches(adata, dataset_name = 'mESCs',
                      y = y.head(2000),
                      val_var = ['Arid1a', 'Atrx', 'Baz1a'])

#%%
