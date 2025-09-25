#!/usr/bin/env python
# update june 16 2025, sorting the last elements of M see getTips() and near return m,M
import sys#
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
        tn1 = n1
        while(tn1 != root):
            path1.append(tn1)
            tn1 = tn1.ancestor
        #print("@",path1)
        tn1 = n2
        while(tn1 != root):
            path2.append(tn1)
            tn1 = tn1.ancestor
        #print("@", path2)
        for i,pi in enumerate(path1[:-1]):
            path1[i+1].age = path1[i].age + path1[i].blength
            #print(i+1, path1[i+1],path1[i+1].age)
        for i,pi in enumerate(path2[:-1]):
            path2[i+1].age = path2[i].age + path2[i].blength
            #print(i+1, path2[i+1],path2[i+1].age)
        root.age = path1[-1].age + path1[-1].blength
        pp = sorted(set(path1) & set(path2), key = path1.index)
        i = len(pp)  #number of branches shared
        s = np.sum([pi.blength for pi in pp])
        #print("sum",s, end=' ')
        if len(pp)>0:
            a = pp[-1].age
        else:
            a = root.age
        return i,s, a
        #path1 = []
        #path2 = []
        #if (not self.findPath(root, path1, n1.name) or not self.findPath(root, path2, n2.name)):
        #    return -1
        #i = 0
        #print("PB",pp)
        #print("TK",[path1,path2])
        #sum = np.zeros(len(path1))
        #while(i < len(path1) and i < len(path2)):
        #    if path1[i] != path2[i]:
        #        break

        #    i += 1
        #    sum[i]= sum[i-1]+(path1[i]).blength 
            #print ('number of edges on path Root-MRCA =', i-1)
            #print ('length of path Root-MRCA =', sum[i-1])
        #print("TKsum",sum[i-1])
        #return(i-1, sum[i-1])


    def PairTips(self,p, tips, PairList):
        newtips = tips.copy()
        newtips.sort(key=lambda x: x.name)
        for i in range(len(newtips)):
            for j in range(i+1, len(newtips)):
                PairList.append([newtips[i], newtips[j]])
        PairList.sort(key=lambda x: (x[0].name, x[1].name))
        return PairList


    import sys
    def MetricsVectors(self, newick,tips,tipslength,PairList):
        m=[]
        M=[]
        a=[]
        inter=[]
        self.myread(newick,self.root)          
        self.getTips(self.root, tips, tipslength)
        tips.sort(key=lambda tip: tip.name)
        #[print(ti.name,ti.ancestor) for ti in tips]
        #sys.exit()
        #print(newick)
        self.PairTips(self.root, tips, PairList)
        for pair in PairList:
            print(f"pair={pair[0].name}-{pair[1].name}", end=' ')
            pop1 = int(pair[0].name.split('_')[0])
            pop2 = int(pair[1].name.split('_')[0])
            #interelement = pop2 - pop1
            interelement = (pop1,pop2)
            #pair[0].debugprint()
            numEdgePath, LengthPath, age  = self.Root_MRCA(self.root, pair[0], pair[1])
            a.append(age)
            inter.append(interelement)
            m.append(numEdgePath)
            M.append(LengthPath)
            print(f"{interelement=},{age=},{numEdgePath=},{LengthPath=} {self.root.age}")
        print(newick)
        ptips=[1] * len(tips)
        #print('ptips=',ptips)
        m=m+ptips
        tipslength.sort() # sort according to name
        M=M+[ti[1] for ti in tipslength]
        return (m, M, a, inter)
    

if __name__ == "__main__":
    m=[]
    M=[]
    age=[]
    inter=[]
    for i in range(2):
        PairList=[]
        tips=[]
        tipslength=[]
        mmtree = mmtree()
        tips=[]
        a = Metrics_Vectors(tree, tips, tipslength, PairList)
        m.append(a[0])
        M.append(a[1])
        age.append(a[2])
        inter.append(a[3])
    print('m=',m ,'\nM=',M, f"{age=}, {inter=}")
