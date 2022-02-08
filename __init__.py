import os
import numpy as np
from zlib import adler32, crc32
import regex as re
from itertools import combinations
from tqdm import tqdm


_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_FINGERPRINT_BATCH_SIZE = int(1e5)
_MIN_CUDA_SIZE = int(1e4)

try:
    import cupy as cp
    _CUDA = True
except:
    print('No cupy installation found. Fingerprinting will use CPU only, which can be slower for large sets.')
    _CUDA = False

try:
    _BIAS_COEF = np.load(os.path.join(_ROOT_DIR,'bias_coef.npy'))
except:
    print('Warning: No bias correction coefficients found for cardinality estimation. Accuracy will be reduced.')
    _BIAS_COEF = None


class VectorizedMinHash():
    '''
    A specialized version of minhash that uses vectorized universal hashing
    to create fingerprints of sets, with cuda support via cupy.

    Hashing implementation based on datasketch MinHash:
    https://ekzhu.github.io/datasketch/_modules/datasketch/minhash.html#MinHash

    'mirror=True' doubles the length of the fingerprint for a given n_perm by taking
    the max of each perm column as well as the min. Saves processing time and
    improves accuracy, while still allowing merging with a min operation.
    '''
    def __init__(self,n_perm=32,mirror=True,seed=1,card_bias_coef=_BIAS_COEF,card_bias_scaler=None):
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

        # Compute bias scaler for cardinality estimate (a function of n_perm)
        if card_bias_scaler is not None:
            self.card_bias_scaler = card_bias_scaler
        elif card_bias_coef is not None:
            self.card_bias_scaler = np.dot(np.array([1,n_perm,np.log(n_perm)]),card_bias_coef)
        else:
            self.card_bias_scaler = None

    def _batch_fingerprint(self,h,cuda='auto'):
        '''
        Takes a sequence of hash values and creates a minHash fingerprint
        of length n_perm or 2*n_perm if mirror=True.
        '''
        h = np.array(h,dtype=np.uint32)[:,np.newaxis]
        a,b = self.permutations

        if cuda == 'auto':
            cuda = _CUDA and (len(h) >= _MIN_CUDA_SIZE)

        if cuda:
            # Pass data to the GPU
            h = cp.asarray(h)
            a = cp.asarray(a)
            b = cp.asarray(b)
            p = cp.asarray(np.uint64(self._mersenne_prime))
            m = cp.asarray(np.uint64(self._max_hash))

            # Run same universal hashing algorithm as cpu version
            H = cp.tile(h,self.n_perm)
            H = cp.bitwise_and((a*H + b) % p, m)

            f = cp.asnumpy(H.min(axis=0))
            if self.mirror:
                f_mirrored = cp.asnumpy(H.max(axis=0))
                f_mirrored = self._max_hash - f_mirrored
                f = np.hstack([f,f_mirrored])

            # Clear gpu cache
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

    def fingerprint(self,h,batch_size=_FINGERPRINT_BATCH_SIZE,cuda='auto'):
        '''
        Computes a fingerprint in batches. Useful if the number of hashes
        is very very high or memory is constrained.
        '''
        fingerprints = [self._batch_fingerprint(h[i:i+batch_size],cuda=cuda) for i in range(0,len(h),batch_size)]

        return union(fingerprints)

    def cardinality(self,fingerprints):
        '''
        Estimate cardinality of set represented by a fingerprint using
        a bias-corrected maximum-likelyhood estimator.
        '''
        m = self._hash_range
        x = np.log(m - fingerprints) - np.log(m)

        if x.ndim > 1:
            c = -1/(np.mean(x,axis=1))
        else:
            c = -1/(np.mean(x))

        if self.card_bias_scaler is not None:
            # Subtract estimated bias
            c -= c*self.card_bias_scaler

        return c


def union(fingerprints):
    '''
    Merge fingerprints to create a new fingerprint. Mathematically equivalent
    to set union. Functionally equivalent output to concatenating hash value
    sequences before fingerprinting.
    '''
    h = np.vstack(fingerprints)
    assert h.shape[0] == len(fingerprints)

    return h.min(axis=0)


def jaccard(f0,f1):
    '''
    Compute the estimated jaccard similarity from two fingerprints
    '''
    return (f0 == f1).mean()


def jaccard_matrix(fingerprints):
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


def jaccard_similarities(query_fingerprints,key_fingerprints=None,cuda='auto'):
    if isinstance(query_fingerprints,np.ndarray):
        query_fingerprints = [query_fingerprints]
    if key_fingerprints is None:
        key_fingerprints = query_fingerprints

    F_q = np.vstack(query_fingerprints)
    F_k = np.vstack(key_fingerprints)
    n_q,n_perm_q = F_q.shape
    n_k,n_perm_k = F_k.shape
    assert n_perm_k == n_perm_q

    if cuda == 'auto':
        cuda = _CUDA and (len(F_k) >= _MIN_CUDA_SIZE)

    if cuda:
        F_k = cp.asarray(F_k)
        F_q = cp.asarray(F_q)

    for f_q in tqdm(F_q,delay=1,desc='Computing Jaccard similarities'):
        jaccard = (F_k == f_q).mean(axis=1)
        if cuda:
            jaccard = cp.asnumpy(jaccard)
        yield jaccard


def jaccard_match(query_fingerprints,key_fingerprints,cuda='auto'):
    keys = np.array(range(len(key_fingerprints)))
    for jaccard in jaccard_similarities(query_fingerprints,key_fingerprints,cuda=cuda):
        yield keys[jaccard == jaccard.max()]


def jaccard_cluster(fingerprints,threshold=0.9,cuda='auto'):
    F = np.vstack(fingerprints)
    n,n_perm = F.shape

    # Initializer clusters as singletons
    ids = np.array(range(n))
    cluster_ids = np.array(range(n))

    if cuda == 'auto':
        cuda = _CUDA and (len(F) >= _MIN_CUDA_SIZE)

    if cuda:
        F = cp.asarray(F)
        ids = cp.asarray(ids)
        cluster_ids = cp.asarray(cluster_ids)
        xp = cp
    else:
        xp = np

    with tqdm(total=n,delay=1,desc='Computing Jaccard clusters') as pbar:
        for i in range(n):
            # Compare each fingerprint i to all fingerprints j >= i and find matches
            c_i = cluster_ids[i]

            # Represent j's as a vector of ids
            j = ids[i:][cluster_ids[i:] != c_i]

            if len(j):
                # Find j's with jaccard > threshold ("matches")
                matched = (F[j] == F[i]).mean(axis=1) >= threshold

                if xp.any(matched):
                    # Get the cluster ids of the matched j's
                    c_j = cluster_ids[j]
                    matched_clusters = xp.unique(c_j[matched])

                    # Identify all fingerprints in these clusters
                    try:
                        ids_to_cluster = xp.isin(cluster_ids,matched_clusters)

                    except cp.cuda.memory.OutOfMemoryError:
                        # In older versions of cupy, isin() can be very memory intensive
                        # (see: https://github.com/cupy/cupy/pull/4018)
                        # Check for cuda out of memory errors, and try moving this step
                        # to the CPU instead.
                        cluster_ids = cp.asnumpy(cluster_ids)
                        matched_clusters = cp.asnumpy(matched_clusters)

                        ids_to_cluster = np.isin(cluster_ids,matched_clusters)

                        cluster_ids = cp.asarray(cluster_ids)
                        matched_clusters = cp.asarray(matched_clusters)

                    # Assign all matched fingerprints to i's group
                    cluster_ids[ids_to_cluster] = c_i

                pbar.update()

            else:
                # No valid j's means that all remaining i's are in the same group already
                # Fill progress bar and exit loop
                pbar.update(n-i)
                break

    if cuda:
        cluster_ids = cp.asnumpy(cluster_ids)
        # Clear gpu cache
        # cp.get_default_memory_pool().free_all_blocks()

    return cluster_ids


def _cut_bytes(b,n=4,offset=0):
    dtypes = {1:np.uint8,2:np.uint16,4:np.uint32}
    end = len(b)-offset
    b_view = np.frombuffer(b,dtype=np.uint8)
    b_view = b_view[offset: end - (end % n) + offset]
    return np.frombuffer(b_view,dtype=dtypes[n])


def byte_hashes(b,n=4):
    """
    Breaks a bytestring into character level n-grams and uses numpy type
    conversion to cast each n-gram as integer.

    n must be equal to 1,2, or 4.
    """
    if not n in [1,2,4]:
        raise ValueError('n must be in [1,2,4]')
    h = np.hstack([cut_bytes(b,n,offset) for offset in range(n)])
    h = np.unique(h)
    return h


def token_hashes(tokens,n=1):
    """
    Converts a sequence of string tokens into ngrams and then hashes each ngram
    using crc32.
    """
    if n == 1:
        ngrams = set(tokens)
    elif n>1:
        ngrams = {'_'.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}
    else:
        raise ValueError('n must be greater than or equal to 1')

    h = np.array([crc32(ngram.encode('utf8')) for ngram in ngrams],dtype=np.uint32)

    return h


#
# def pattern_hashes(s,pattern,flags=0,overlapped=False):
#     """
#     Breaks a string into shingles that match a regex pattern, and uses the
#     adler32 algorithm to hash each shingle.
#     """
#     h = np.array([adler32(t.group(0).encode('utf8')) for t in re.finditer(pattern,s,flags=flags,overlapped=overlapped)],dtype=np.uint32)
#     return h
#
#
# def token_hashes(s,n=2):
#     """
#     Breaks a string into whitespace-separated token n-grams and uses the
#     adler32 algorithm to hash each n-gram.
#     """
#     s = re.sub('\s+',' ',s)
#     pattern = fr'(\b\S+\s*){{{n}}}'
#     overlapped = n > 1
#
#     return pattern_hashes(s,pattern=pattern,overlapped=overlapped)
