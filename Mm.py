#!/usr/bin/env python
# update june 16 2025, sorting the last elements of M see getTips() and near return m,M
#
import tree
from tree import Tree, Node
import numpy as np

class mmtree(Tree):
    i = 0
    def __init__(self):        
        super().__init__()
        #print(Tree.__module__)
        #print(self.root)
        
    def printTiplabels(self,p):
        if not(tree.istip(p)):
            self.printTiplabels(p.left)
            self.printTiplabels(p.right)
        else:
            print (p.name)
        
            
    def getTips(self,p, tips, tipslength):
        if not(tree.istip(p)):
            self.getTips(p.left, tips, tipslength)
            self.getTips(p.right, tips, tipslength)
        else:
            tips.append(p)
            tipslength.append([p.name, p.blength])


    def findPath(self, root, path, n):
            if root is None:
                return False

            path.append(root)
            if root.name == n :
                return True

            if ((root.left != -1 and self.findPath( root.left, path, n)) or
                    (root.right!= -1 and self.findPath( root.right, path, n))):
                return True

            path.pop()
            return False

    def Root_MRCA(self, root, n1, n2):
        path1 = []
        path2 = []

        if (not self.findPath(root, path1, n1.name) or not self.findPath(root, path2, n2.name)):
            return -1
        i = 0
        sum = np.zeros(len(path1))
        while(i < len(path1) and i < len(path2)):
            if path1[i] != path2[i]:
                break

            i += 1
            sum[i]= sum[i-1]+(path1[i]).blength 
            #print ('number of edges on path Root-MRCA =', i-1)
            #print ('length of path Root-MRCA =', sum[i-1])
        return(i-1, sum[i-1])


    def PairTips(self,p, tips, PairList):
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                PairList.append([tips[i], tips[j]])  
        return PairList


    import sys
    def MetricsVectors(self, newick,tips,tipslength,PairList):
        m=[]
        M=[]
        self.myread(newick,self.root)          
        self.getTips(self.root, tips, tipslength)
        tips.sort(key=lambda tip: tip.name)
        #[ print(ti.name) for ti in tips]
        #sys.exit()
        self.PairTips(self.root, tips, PairList)
        for pair in PairList:
            numEdgePath, LengthPath = self.Root_MRCA(self.root, pair[0], pair[1]) 
            m.append(numEdgePath)
            M.append(LengthPath)
        ptips=[1] * len(tips)
        #print('ptips=',ptips)
        m=m+ptips
        tipslength.sort() # sort according to name
        M=M+[ti[1] for ti in tipslength]
        return (m, M)
    

if __name__ == "__main__":
    m=[]
    M=[]
    for i in range(2):
        PairList=[]
        tips=[]
        tipslength=[]
        mmtree = mmtree()
        tips=[]
        a = Metrics_Vectors(tree, tips, tipslength, PairList)
        m.append(a[0])
        M.append(a[1])
    print('m=',m ,'\nM=',M)
