# This script is not used currently. It is for future reference.
# source: https://github.com/habedi/-myPython-LanguageCodes/blob/master/codes/personalized_pagerank.py

import networkx as nx
from operator import itemgetter
from collections import OrderedDict


def ppr(g, seed, alpha=0.85, epsilon=10e-8, iters=100):
    pref = {}
    T = [seed]
    for node in g.neighbors(seed):
        T.append(node)
    for node in g:
        if node in T:
            pref.update({node: (1.0 / len(T))})
        else:
            pref.update({node: 0.0})
    return nx.pagerank(
        g, alpha=alpha, personalization=pref, max_iter=iters, tol=epsilon)


def ppr_sorted(g, pprv):
    spprv = {}
    for item in pprv.iteritems():
        k, v = item
        spprv.update({k: (v / g.degree(k))})
        pass
    return sorted(spprv.items(), key=itemgetter(1), reverse=True)


def min_cond_cut(g, dspprv, max_cutsize=0):

    def conductance(nbunch):
        # print nbunch
        sigma = 0.0
        vol1 = vol2 = 0
        for node in nbunch:
            for n in g.neighbors(node):
                if n not in nbunch:
                    sigma += 1.0
        for degseq in g.degree().iteritems():
            node, degree = degseq
            if node not in nbunch:
                vol2 += degree
            else:
                vol1 += degree
        # print nbunch
        return (sigma / min(vol1, vol2))

    k = 1
    conductance_list = []
    if max_cutsize < 1:
        # cutsize could be as big as the graph itself
        limit = (len(dspprv))
    else:
        # maximum size of the cut with minimum conductance
        limit = max_cutsize
    while k < limit:
        nbunch = []
        for i in xrange(0, k):
            nbunch.append(dspprv[i][0])
        c = (k, conductance(nbunch))
        # conductane of current cut size
        conductance_list.append(c)
        k += 1
    return min(conductance_list, key=itemgetter(1))


## running the code ..
def loadGraph(gfile):
    return nx.read_edgelist(
        path=gfile, comments='#', delimiter="\t", nodetype=int)


g = loadGraph(gfile='sample.txt')
a = ppr(g, seed=0, alpha=0.85, epsilon=10e-8, iters=100)
# print a
# exit()
b = ppr_sorted(g, pprv=a)
# finding the best community around NodeId==5
print(min_cond_cut(g, dspprv=b, max_cutsize=6))
