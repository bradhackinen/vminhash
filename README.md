# vectorizedMinHash
A small toolkit for very efficiently comparing the similarity of large numbers of documents or other data structures that can be represented as sets. Core features:
- Very fast construction of MinHash "fingerprints" of sets. The algorithm is inspired by the MinHash implementation in [datasketch](https://github.com/ekzhu/datasketch), but the core MinHash algorithm is vectorized in numpy and includes CUDA support via [cupy](https://cupy.chainer.org/).
- Jaccard similarity estimation
- Cardinality estimation (with bias correction for much better accuracy)
- Union operations between fingerprints

The module also includes some simple functions for quickly converting text to token or n-gram-based hashes.

## Requirements
- numpy
- cupy and CUDA for GPU acceleration (optional)

## Installation
I haven't made it pip installable yet, so you'll just have to grab the repository files for now.

## How to use
### Generating fingerprints
The module is built around a `VectorizedMinHash` hasher object. To construct fingerprint from a string using CUDA support:
```python
from vectorizedMinHash import VectorizedMinHash,fastNGramHashes

hasher = VectorizedMinHash(n_perm=256,mirror=True)


testString = 'This is a test string'

# 1) Construct a fingerprint from n-gram features
# (fastNGramHashes converts bytes directly to n-gram ids by changing the stride of the dtype)
ngramHashes = fastNGramHashes(testString.encode('ascii'),n=4)
fingerprint = hasher.fingerprint(ngramHashes,cuda=True)

# 2) Construct a fingerprint from tokens (split on whitespace by default)
tokenHashes = tokenHashes(testString.encode('ascii'))
fingerprint = hasher.fingerprint(ngramHashes,cuda=True)

# 3) Construct a fingerprint from tokens (split on non-alphanumeric chars)
tokenHashes = tokenHashes(testString.encode('ascii'),tokenRE=rb'\w+')
fingerprint = hasher.fingerprint(tokenHashes,cuda=True)
```
You can also compute a fingerprint from a more generic array of integer ids (must be representable as a `np.uint32` array)
```python
# Construct a fingerprint from element ids
ids = range(10)
fingerprint = hasher.fingerprint(ids,cuda=True)
```
The constructor's most important parameter is `n_perm`, which sets the size of the fingerprints. Larger fingerprints are more accurate, but require more processing time and memory to store. `mirror` doubles the length of the fingerprint for a given `n_perm` by using the max operation as in addition to min. In my experiments this saves processing time and improves accuracy, so it is the default setting.

A fingerprint is a simple `np.uint32` array. It doesn't remember what hasher made it, so __be careful to only compare fingerprints made with exactly the same settings__.

### Merging fingerprints with union
Fingerprints can be merged with the `union` function. This operation is equivalent to taking the union (or concatenating all the hash values) of the input sets before fingerprinting. Behind the scenes, it just stacks the fingerprints into an 2-d array and takes the min.
```python
from vectorizedMinHash import union

new_fingerprint = union([fingerprint0,fingerprint1,fingerprint2])
```
Fingerprints constructed via `union` are indistinguishable from other fingerprints and it is perfectly valid to use them in jaccard or cardinality estimates.

### Jaccard similarity index
The Jaccard similarity index between two sets can be estimated by the fraction of equal values in the corresponding fingerprints. The module contains two helper functions: `jaccard` and `jaccardMatrix`:
```python
from vectorizedMinHash import jaccard, jaccardMatrix

# Compute jaccard similarity index for two fingerprints:
jac = jaccard(fingerprint0,fingerprint1)

# Compute all pairwise similarities for three fingerprints
jac_matrix = jaccardMatrix([fingerprint0,fingerprint1,fingerprint2])
```
### cardinality estimates
The cardinality of a fingerprint (that is, the number of distinct hashes used to generate it) can be estimated with the `cardinality` method:
```python
n = hasher.cardinality(fingerprint)
```
#### A Note on bias correction
Accurately estimating the cardinality of a set from its fingerprint is a little tricky. The method used in Datasketch MinHash has a huge downward bias, and more accurate methods usually involve completely different hashing algorithms (which can't simultaneously compute Jaccard similarity). The implementation in this module uses a simple bias-corrected maximum likelihood estimator to significantly increase the accuracy. The only complication is that the bias correction must be calculated empirically. The module has the required code at the end of the `__init__.py` file. Pre-computed bias correction coefficients are stored in `bias_coef.npy`, and should load automatically.
