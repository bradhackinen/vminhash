import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from vectorizedMinHash import *
import time
from unidecode import unidecode


def jaccardAccuracyTest(baseBytes,n_fractions=501):

    randBytes = bytes(random.getrandbits(8) for _ in range(len(baseBytes)))

    randFractions = np.linspace(0,0.5,n_fractions)
    mixtures = [bytes(b1 if random.uniform(0,1) < f else b0 for b0,b1 in zip(baseBytes,randBytes))
                for f in randFractions]

    results = []

    #Compute correct jaccard distances
    true_jaccard = {}
    s0 = set(list(fastNGramHashes(baseBytes)))
    for f,mixtureBytes in zip(randFractions,mixtures):
        s1 = set(list(fastNGramHashes(mixtureBytes)))
        true_jaccard[f] = len(s0 & s1) / len(s0 | s1)

    for cuda in False,True:
        for mirror in False,True:
            for n_perm in [128,256,512,1024]:
                mh = vectorizedMinHash(n_perm,mirror=mirror)

                h0 = mh.fingerprint(fastNGramHashes(baseBytes,batch_size=1000),cuda=cuda)
                for f,mixtureBytes in zip(randFractions,mixtures):

                    start_time = time.time()
                    h1 = mh.fingerprint(fastNGramHashes(mixtureBytes),cuda=cuda)
                    end_time = time.time()

                    jac = jaccard(h0,h1)
                    results.append({'cuda':cuda,'n_perm':n_perm,'mirror':mirror,'rand_fraction':f,'jaccard':jac,'time':end_time-start_time})

    df = pd.DataFrame(results)
    df['true_jaccard'] = df['rand_fraction'].replace(true_jaccard)
    df['error'] = np.abs(df['jaccard'] - df['true_jaccard'])

    return df


sampleText = '''
MinHash
From Wikipedia, the free encyclopedia
In computer science and data mining, MinHash (or the min-wise independent permutations locality sensitive hashing scheme) is a technique for quickly estimating how similar two sets are. The scheme was invented by Andrei Broder (1997),[1] and initially used in the AltaVista search engine to detect duplicate web pages and eliminate them from search results.[2] It has also been applied in large-scale clustering problems, such as clustering documents by the similarity of their sets of words.[1]

Contents
1	Jaccard similarity and minimum hash values
2	Algorithm
2.1	Variant with many hash functions
2.2	Variant with a single hash function
2.3	Time analysis
3	Min-wise independent permutations
4	Applications
5	Other uses
6	Evaluation and benchmarks
7	See also
8	References
Jaccard similarity and minimum hash values
The Jaccard similarity coefficient is a commonly used indicator of the similarity between two sets. For sets A and B it is defined to be the ratio of the number of elements of their intersection and the number of elements of their union:

{\displaystyle J(A,B)={{|A\cap B|} \over {|A\cup B|}}.} J(A,B)={{|A\cap B|} \over {|A\cup B|}}.
This value is 0 when the two sets are disjoint, 1 when they are equal, and strictly between 0 and 1 otherwise. Two sets are more similar (i.e. have relatively more members in common) when their Jaccard index is closer to 1. The goal of MinHash is to estimate J(A,B) quickly, without explicitly computing the intersection and union.

Let h be a hash function that maps the members of A and B to distinct integers, and for any set S define hmin(S) to be the minimal member of S with respect to h—that is, the member x of S with the minimum value of h(x). Now, applying hmin to both A and B, and assuming no hash collisions, we will get the same value exactly when the element of the union A ∪ B with minimum hash value lies in the intersection A ∩ B. The probability of this being true is the ratio above, and therefore:

Pr[ hmin(A) = hmin(B) ] = J(A,B),
That is, the probability that hmin(A) = hmin(B) is true is equal to the similarity J(A,B), assuming randomly chosen sets A and B. In other words, if r is the random variable that is one when hmin(A) = hmin(B) and zero otherwise, then r is an unbiased estimator of J(A,B). r has too high a variance to be a useful estimator for the Jaccard similarity on its own, because {\displaystyle r} r is always zero or one. The idea of the MinHash scheme is to reduce this variance by averaging together several variables constructed in the same way.

Algorithm
Variant with many hash functions
The simplest version of the minhash scheme uses k different hash functions, where k is a fixed integer parameter, and represents each set S by the k values of hmin(S) for these k functions.

To estimate J(A,B) using this version of the scheme, let y be the number of hash functions for which hmin(A) = hmin(B), and use y/k as the estimate. This estimate is the average of k different 0-1 random variables, each of which is one when hmin(A) = hmin(B) and zero otherwise, and each of which is an unbiased estimator of J(A,B). Therefore, their average is also an unbiased estimator, and by standard deviation for sums of 0-1 random variables, its expected error is O(1/√k).[3]

Therefore, for any constant ε > 0 there is a constant k = O(1/ε2) such that the expected error of the estimate is at most ε. For example, 400 hashes would be required to estimate J(A,B) with an expected error less than or equal to .05.

Variant with a single hash function
It may be computationally expensive to compute multiple hash functions, but a related version of MinHash scheme avoids this penalty by using only a single hash function and uses it to select multiple values from each set rather than selecting only a single minimum value per hash function. Let h be a hash function, and let k be a fixed integer. If S is any set of k or more values in the domain of h, define h(k)(S) to be the subset of the k members of S that have the smallest values of h. This subset h(k)(S) is used as a signature for the set S, and the similarity of any two sets is estimated by comparing their signatures.

Specifically, let A and B be any two sets. Then X = h(k)(h(k)(A) ∪ h(k)(B)) = h(k)(A ∪ B) is a set of k elements of A ∪ B, and if h is a random function then any subset of k elements is equally likely to be chosen; that is, X is a simple random sample of A ∪ B. The subset Y = X ∩ h(k)(A) ∩ h(k)(B) is the set of members of X that belong to the intersection A ∩ B. Therefore, |Y|/k is an unbiased estimator of J(A,B). The difference between this estimator and the estimator produced by multiple hash functions is that X always has exactly k members, whereas the multiple hash functions may lead to a smaller number of sampled elements due to the possibility that two different hash functions may have the same minima. However, when k is small relative to the sizes of the sets, this difference is negligible.

By standard Chernoff bounds for sampling without replacement, this estimator has expected error O(1/√k), matching the performance of the multiple-hash-function scheme.

Time analysis
The estimator |Y|/k can be computed in time O(k) from the two signatures of the given sets, in either variant of the scheme. Therefore, when ε and k are constants, the time to compute the estimated similarity from the signatures is also constant. The signature of each set can be computed in linear time on the size of the set, so when many pairwise similarities need to be estimated this method can lead to a substantial savings in running time compared to doing a full comparison of the members of each set. Specifically, for set size n the many hash variant takes O(n k) time. The single hash variant is generally faster, requiring O(n) time to maintain the queue of minimum hash values assuming n >> k.[1]

Min-wise independent permutations
In order to implement the MinHash scheme as described above, one needs the hash function h to define a random permutation on n elements, where n is the total number of distinct elements in the union of all of the sets to be compared. But because there are n! different permutations, it would require Ω(n log n) bits just to specify a truly random permutation, an infeasibly large number for even moderate values of n. Because of this fact, by analogy to the theory of universal hashing, there has been significant work on finding a family of permutations that is "min-wise independent", meaning that for any subset of the domain, any element is equally likely to be the minimum. It has been established that a min-wise independent family of permutations must include at least

{\displaystyle \operatorname {lcm} (1,2,\cdots ,n)\geq e^{n-o(n)}} \operatorname {lcm} (1,2,\cdots ,n)\geq e^{n-o(n)}
different permutations, and therefore that it needs Ω(n) bits to specify a single permutation, still infeasibly large.[2]

Because of this impracticality, two variant notions of min-wise independence have been introduced: restricted min-wise independent permutations families, and approximate min-wise independent families. Restricted min-wise independence is the min-wise independence property restricted to certain sets of cardinality at most k.[4] Approximate min-wise independence has at most a fixed probability ε of varying from full independence.[5]

Applications
The original applications for MinHash involved clustering and eliminating near-duplicates among web documents, represented as sets of the words occurring in those documents.[1][2][6] Similar techniques have also been used for clustering and near-duplicate elimination for other types of data, such as images: in the case of image data, an image can be represented as a set of smaller subimages cropped from it, or as sets of more complex image feature descriptions.[7]

In data mining, Cohen et al. (2001) use MinHash as a tool for association rule learning. Given a database in which each entry has multiple attributes (viewed as a 0–1 matrix with a row per database entry and a column per attribute) they use MinHash-based approximations to the Jaccard index to identify candidate pairs of attributes that frequently co-occur, and then compute the exact value of the index for only those pairs to determine the ones whose frequencies of co-occurrence are below a given strict threshold.[8]

The MinHash algorithm has been adapted for bioinformatics, where the problem of comparing genome content has a similar theoretical underpinning to that of comparing documents on the web. There are various software implementations for this, including mash [9] and sourmash [10]. These tools allow the very rapid comparison of whole genome sequencing data with reference genomes (around 3 minutes to compare one genome with the 90000 reference genomes in RefSeq), and are suitable for speciation and maybe a limited degree of microbial sub-typing. There are also applications for metagenomics [9] and the use of MinHash derived algorithms for genome alignment and genome assembly[11].

Other uses
The MinHash scheme may be seen as an instance of locality sensitive hashing, a collection of techniques for using hash functions to map large sets of objects down to smaller hash values in such a way that, when two objects have a small distance from each other, their hash values are likely to be the same. In this instance, the signature of a set may be seen as its hash value. Other locality sensitive hashing techniques exist for Hamming distance between sets and cosine distance between vectors; locality sensitive hashing has important applications in nearest neighbor search algorithms.[12] For large distributed systems, and in particular MapReduce, there exist modified versions of MinHash to help compute similarities with no dependence on the point dimension.[13]

Evaluation and benchmarks
A large scale evaluation has been conducted by Google in 2006 [14] to compare the performance of Minhash and SimHash[15] algorithms. In 2007 Google reported using Simhash for duplicate detection for web crawling[16] and using Minhash and LSH for Google News personalization.[17]
'''


resultsDF = accuracyTest(unidecode(sampleText).encode('ascii'))


resultsDF.groupby('cuda')['error'].mean()

sb.lmplot(x='true_jaccard',y='jaccard',data=resultsDF[resultsDF['cuda']],hue='n_perm',col='mirror',fit_reg=False,size=4,aspect=1)
plt.plot([0,1],[0,1],lw=0.5,c='k')

sb.lmplot(x='n_perm',y='error',data=resultsDF[resultsDF['cuda']].groupby(['mirror','n_perm']).mean().reset_index(),hue='mirror',fit_reg=False,size=4,aspect=1)
plt.plot([0,1],[0,1],lw=0.5,c='k')
plt.gca().set_ylim((0,None))

sb.lmplot(x='n_perm',y='time',data=resultsDF.groupby(['cuda','n_perm']).mean().reset_index(),hue='cuda',fit_reg=False,size=4,aspect=1)
plt.plot([0,1],[0,1],lw=0.5,c='k')
plt.gca().set_ylim((0,None))





import matplotlib.pyplot as plt
import seaborn as sb

resultsDF = pd.DataFrame()
for n_perm in (64,128,512,1024):
    vectorizedMinHash = VectorizedMinHash(n_perm,mirror=True)

    cardinalities = np.geomspace(10,1e6,num=10).astype(int)
    estimates = np.zeros(len(cardinalities))

    for i,c in enumerate(cardinalities):
        h = np.array(range(c),dtype=np.uint32)
        f = vectorizedMinHash.fingerprint(h,cuda=True)
        estimates[i] = vectorizedMinHash.cardinality(f)

        # Datasketch formula:
        # estimates[i] = np.float(n_perm) / np.sum(f / np.float(vectorizedMinHash._max_hash)) - 1.0


    df = pd.DataFrame(np.vstack([cardinalities.astype(float),estimates]).T,columns=['cardinality','estimate'])
    df['n_perm'] = n_perm
    resultsDF = resultsDF.append(df)

for c in ['cardinality','estimate']:
    resultsDF['log_'+c] = np.log10(resultsDF[c])

sb.lmplot(x='log_cardinality',y='log_estimate',hue='n_perm',data=resultsDF,fit_reg=False)
plt.plot([0,resultsDF['log_cardinality'].max()],[0,resultsDF['log_cardinality'].max()],lw=0.5,c='k')


sb.lmplot(x='cardinality',y='estimate',hue='n_perm',data=resultsDF,fit_reg=False)
plt.plot([0,resultsDF['cardinality'].max()],[0,resultsDF['cardinality'].max()],lw=0.5,c='k')
