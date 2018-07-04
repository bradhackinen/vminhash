import os
import numpy as np
from zlib import adler32
import re
from vectorizedMinHash.definitions import ROOT_DIR

try:
    import cupy as cp
except:
    print('Warning: No cupy installation found. cuda=True will raise an error.')

try:
    bias_coef = np.load(os.path.join(ROOT_DIR,'bias_coef.npy'))
except:
    print('Warning: No bias correction coefficients found for cardinality estimation. Accuracy will be reduced.')
    bias_coef = None


class VectorizedMinHash():
    '''
    A specialized version of minhash that uses vectorized universal hashing
    to create fingerprints of sets, with cuda support via cupy.

    Hashing implementation based on datasketch MinHash:
    https://ekzhu.github.io/datasketch/_modules/datasketch/minhash.html#MinHash

    'Mirror' doubles the length of the fingerprint for a given n_perm by taking
    the max of each perm column as well as the min. Saves processing time and
    improves accuracy, while still allowing merging with a min operation.
    '''

    def __init__(self,n_perm=64,mirror=True,seed=1,card_bias_scaler=None,card_bias_coef=bias_coef):
        self.n_perm = n_perm
        self.mirror = mirror
        self.seed = seed

        self._mersenne_prime = (1 << 61) - 1
        self._max_hash = (1 << 32) - 1
        self._hash_range = (1 << 32)

        self.generator = np.random.RandomState(self.seed)
        # Create parameters for a random bijective permutation function
        # that maps a 32-bit hash value to another 32-bit hash value.
        # http://en.wikipedia.org/wiki/Universal_hashing
        self.permutations = np.array([(self.generator.randint(1, self._mersenne_prime, dtype=np.uint64),
                                       self.generator.randint(0, self._mersenne_prime, dtype=np.uint64))
                                      for _ in range(self.n_perm)], dtype=np.uint64).T

        #Compute bias scaler for cardinality estimate (a function of n_perm)
        if card_bias_scaler is not None:
            self.card_bias_scaler = bias_scaler
        elif card_bias_coef is not None:
            self.card_bias_scaler = np.dot(np.array([1,n_perm,np.log(n_perm)]),card_bias_coef)
        else:
            self.card_bias_scaler = None

    def batchFingerprint(self,h,cuda=True):
        '''
        Takes a sequence of hash values and creates a minHash fingerprint
        of length n_perm or 2*n_perm if mirror=True.
        '''
        h = np.array(h,dtype=np.uint32)[:,np.newaxis]
        a,b= self.permutations

        if cuda:
            #Pass data to the GPU
            h = cp.asarray(h)
            a = cp.asarray(a)
            b = cp.asarray(b)
            p = cp.asarray(np.uint64(self._mersenne_prime))
            m = cp.asarray(np.uint64(self._max_hash))

            #Run same universal hashing algorithm as cpu version
            H = cp.tile(h,self.n_perm)
            H = cp.bitwise_and((a*H + b) % p, m)

            f = cp.asnumpy(H.min(axis=0))
            if self.mirror:
                f_mirrored = cp.asnumpy(H.max(axis=0))
                f_mirrored = self._max_hash - f_mirrored
                f = np.hstack([f,f_mirrored])

            #Clear gpu cache
            cp.get_default_memory_pool().free_all_blocks()

        else:
            H = np.tile(h,self.n_perm)
            H = np.bitwise_and((a*H + b) % self._mersenne_prime, np.uint64(self._max_hash))
            f = H.min(axis=0)

            if self.mirror:
                f_mirrored = H.max(axis=0)
                f_mirrored = self._max_hash - f_mirrored
                f = np.hstack([f,f_mirrored])

        return f.astype(np.uint32)



    def fingerprint(self,h,batch_size=1000,cuda=True):
        if not len(h):
            raise Exception('Cannot fingerprint zero-length hash array')
        '''
        Computes a fingerprint in batches. Useful if the number of hashes
        is very very high or memory is constrained.
        '''
        fingerprints = [self.batchFingerprint(h[i:i+batch_size],cuda=cuda) for i in range(0,len(h),batch_size)]
        return union(fingerprints)


    def cardinality(self,fingerprints,bias_coef=bias_coef):
        '''
        Estimate cardinality of set represented by a fingerprint using
        a bias-corrected maximum-likelyhood estimator.
        '''
        M = self._hash_range
        x = np.log(M - fingerprints) - np.log(M)

        if x.ndim > 1:
            c =  -1/(np.mean(x,axis=1))
        else:
            c = -1/(np.mean(x))

        if self.card_bias_scaler is not None:
            #Subtract estimated bias
            c -= c*self.card_bias_scaler

        return c


def union(fingerprints):
    '''
    Merge two fingerprints to create a new fingerprint. Mathematically equivalent
    to set union. Functionally equivalent output to concatenating hash value
    sequences before fingerprinting.
    '''
    H = np.vstack(fingerprints)
    return H.min(axis=0)


def jaccard(f0,f1):
    '''
    Compute the estimated jaccard similarity from two fingerprints
    '''
    return (f0 == f1).mean()


def jaccardMatrix(fingerprints):
    F = np.vstack(fingerprints)
    n = F.shape[0]
    X = np.zeros((n,n))
    for i in range(n):
        x_i = (F[i:,:] == F[:n-i,:]).mean(axis=1)

        #Fill upper triangle
        np.fill_diagonal(X[i:,:n-i],x_i)

        #Fill lower triangle
        np.fill_diagonal(X[:n-i,i:],x_i)
    return X


def cutBytes(b,n=4,offset=0):
    dtypes = {1:np.uint8,2:np.uint16,4:np.uint32}
    if not n in dtypes:
        raise Exception('n must be in [2,4]')
    end = len(b)-offset
    b_view = np.frombuffer(b,dtype=np.uint8)
    b_view = b_view[offset: end - (end % n) + offset]
    return np.frombuffer(b_view,dtype=dtypes[n])


def fastNGramHashes(b,n=4):
    h = np.hstack([cutBytes(b,n,offset) for offset in range(n)])
    h = np.unique(h)
    return h

def tokenHashes(b,tokenRE=rb'\S+'):
    h = np.array([adler32(t) for t in re.findall(tokenRE,b)],dtype=np.uint32)
    return h





if __name__ == '__main__':

    #Compute bias correction coeffients
    import os
    import pandas as pd
    import statsmodels.api as sm

    n_samples = 1000
    min_card = 10
    max_card = int(1e6)
    min_perm = 10
    max_perm = 1000

    biasDF = pd.DataFrame(index=range(n_samples))
    biasDF['cardinality'] = np.random.uniform(min_card,max_card,size=n_samples).astype(int)
    biasDF['n_perm'] = np.random.uniform(min_perm,max_perm,size=n_samples).astype(int)

    estimates = []
    for i,c,n_perm in biasDF.itertuples():
        vectorizedMinHash = VectorizedMinHash(n_perm)
        h = np.array(range(c),dtype=np.uint32)
        f = vectorizedMinHash.fingerprint(h,cuda=True)

        estimates.append(vectorizedMinHash.cardinality(f))

    biasDF['estimate'] = estimates

    biasDF['bias'] = biasDF['estimate'] - biasDF['cardinality']

    biasDF['log_n_perm'] = np.log(biasDF['n_perm'])
    biasDF['constant'] = 1

    X = biasDF[['estimate','n_perm','log_n_perm']].multiply(biasDF['estimate'],axis=0)
    ols = sm.OLS(biasDF['bias'],X)

    print(ols.fit().summary())

    bias_coef = ols.fit().params

    np.save(os.path.join(ROOT_DIR,'bias_coef.npy'),bias_coef.values)
