import numpy as np
import cupy as cp
from zlib import adler32
import regex as re

class VectorizedMinHash():
    '''
    A specialized version of minhash that uses vectorized universal hashing
    to create fingerprints, with cuda support via cupy.

    Hashing implementation based on datasketch MinHash:
    https://ekzhu.github.io/datasketch/_modules/datasketch/minhash.html#MinHash

    'Mirror' doubles the length of the fingerprint for a given n_perm by taking
    the max of each perm column as well as the min. Saves processing time and
    improves accuracy.
    '''

    def __init__(self,n_perm=64,mirror=True,seed=1):
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



    def batchFingerprint(self,h,cuda=True):
        '''
        Takes a sequence of hash values and creates a minHash fingerprint
        of length n_perm or 2*n_perm if mirror=True.
        '''
        h = np.array(h)[:,np.newaxis]
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
                f = np.hstack([f,cp.asnumpy(H.max(axis=0))])

            #Clear gpu cache
            cp.get_default_memory_pool().free_all_blocks()

        else:
            H = np.tile(h,self.n_perm)
            H = np.bitwise_and((a*H + b) % self._mersenne_prime, np.uint64(self._max_hash))
            f = H.min(axis=0)
            if self.mirror:
                f = np.hstack([f,H.max(axis=0)])

        return f


    def mergeFingerprints(self,fingerprints):
        '''
        Merge two fingerprints to create a new fingerprint. Equivalent output to
        concatenating hash value sequences before fingerprinting.
        '''
        H = np.vstack(fingerprints)
        if self.mirror:
            return np.hstack([H[:,:self.n_perm].min(axis=0),H[:,self.n_perm:].max(axis=0)])
        else:
            return H.min(axis=0)


    def fingerprint(self,h,batch_size=10000,cuda=True):
        if not len(h):
            raise Exception('Cannot fingerprint zero-length hash array')
        '''
        Computes a fingerprint in batches. Useful only if the number of hashes
        is very very high or memory is constrained.
        '''
        fingerprints = [self.batchFingerprint(h[i:i+batch_size],cuda=cuda) for i in range(0,len(h),batch_size)]
        return self.mergeFingerprints(fingerprints)



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

#
# testBytes = b'''MinHash
# From Wikipedia, the free encyclopedia
# In computer science and data mining, MinHash (or the min-wise independent permutations locality sensitive hashing scheme) is a technique for quickly estimating how similar two sets are. The scheme was invented by Andrei Broder (1997),[1] and initially used in the AltaVista search engine to detect duplicate web pages and eliminate them from search results.[2] It has also been applied in large-scale clustering problems, such as clustering documents by the similarity of their sets of words.[1]
# '''
#
# hasher = VectorizedMinHash()
# hasher.fingerprint(fastNGramHashes(testBytes))
#
#
# hasher.fingerprint(tokenHashes(testBytes))
