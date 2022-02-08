import vminhash as vmh
import numpy as np



# Run a vey minimal test
strings = [
            'This is a test string',
            'This is also a test string',
            'This is something else',
            'A fairly different string',
            'completely unrelated text'
            ]

min_hash = vmh.VectorizedMinHash()

f = [min_hash.fingerprint(vmh.token_hashes(s.split(),n=1)) for s in strings]

vmh.jaccard(f[0],f[-1])
vmh.jaccard_matrix(f)
vmh.jaccard_cluster(f,threshold=0.85)

list(vmh.jaccard_similarities(f[:3],f[2:]))

list(vmh.jaccard_match(f[:3],f[2:]))

query_fingerprints = f[:3]
key_fingerprints = f[3:]
keys = np.array(range(len(key_fingerprints)))
for jaccard in jaccard_similarities(query_fingerprints,key_fingerprints,cuda=cuda):
    yield keys[jaccard == jaccard.max()]



"""
Cuda speed test for clustering a large number of strings
"""

strings = strings*20000
f = [min_hash.fingerprint(vmh.token_hashes(s.split(),n=1)) for s in strings]

cluster_ids_cpu = vmh.jaccard_cluster(f,threshold=0.85,cuda=False)

cluster_ids_cuda = vmh.jaccard_cluster(f,threshold=0.85,cuda=True)





"""
Test the accuracy of the jaccard estimation on (moderately) large sets
"""

import vminhash as vmh
import numpy as np
import pandas as pd
import matplotlib as plt


n_tokens = 10000

results = []
for mirror in False,True:
    for i in range(1001):
        overlap = i/1000

        n_intersect = int(n_tokens*overlap)
        n_union = 2*n_tokens - n_intersect
        # base_tokens = np.random.randint(0,t_max,n_union)
        t0 = np.random.randint(0,2**30)
        base_tokens = list(range(t0,t0+n_union))
        doc_tokens = [
                base_tokens[:n_tokens],
                base_tokens[-n_tokens:]
            ]

        strings = [
            ' '.join(str(t) for t in base_tokens[:n_tokens]),
            ' '.join(str(t) for t in base_tokens[-n_tokens:])
            ]

        true_jaccard = n_intersect/n_union

        min_hash = vmh.VectorizedMinHash(n_perm=64,mirror=mirror)
        f = [min_hash.fingerprint(vmh.token_hashes(s.split(),n=1)) for s in strings]
        # f = [min_hash.fingerprint(tokens) for tokens in doc_tokens]

        est_jaccard = vmh.jaccard(f[0],f[1])

        results.append({'true_jaccard':true_jaccard,'est_jaccard':est_jaccard,'mirror':mirror})



results_df = pd.DataFrame(results)
results_df['error'] = (results_df['est_jaccard'] - results_df['true_jaccard'])**2
results_df.groupby('mirror')['error'].mean()


results_df[~results_df['mirror']].plot.scatter(x='true_jaccard',y='est_jaccard',s=1,label='mirror=False')
results_df[results_df['mirror']].plot.scatter(x='true_jaccard',y='est_jaccard',s=1,label='mirror=True')
