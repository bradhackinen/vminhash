import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

from vminhash import VectorizedMinHash,_ROOT_DIR

n_samples = 1000
min_card = 10
max_card = int(1e6)
min_perm = 10
max_perm = 1000

bias_df = pd.DataFrame(index=range(n_samples))
bias_df['cardinality'] = (10**np.random.uniform(np.log10(min_card),np.log10(max_card),size=n_samples)).astype(int)
bias_df['n_perm'] = np.random.uniform(min_perm,max_perm,size=n_samples).astype(int)

print('Estimating cardinality on artificial data')
estimates = []
for c,n_perm in tqdm(bias_df[['cardinality','n_perm']].values):
    vectorizedMinHash = VectorizedMinHash(n_perm)
    h = np.array(range(c),dtype=np.uint32)
    f = vectorizedMinHash.fingerprint(h)

    estimates.append(vectorizedMinHash.cardinality(f))

bias_df['estimate'] = estimates
bias_df['bias'] = bias_df['estimate'] - bias_df['cardinality']
bias_df['log_n_perm'] = np.log(bias_df['n_perm'])
bias_df['constant'] = 1

print('Estimating bias coefficients')
X = bias_df[['estimate','n_perm','log_n_perm']].multiply(bias_df['estimate'],axis=0)
ols = sm.OLS(bias_df['bias'],X)

print(ols.fit().summary())

save_file = os.path.join(_ROOT_DIR,'bias_coef.npy')
np.save(save_file,bias_coef.values)

print(f'Bias coefficients saved as {save_file}')
