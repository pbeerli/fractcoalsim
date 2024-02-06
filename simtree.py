#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/23 11:47    
# @Author  : Peter Beerli
# @Site    : Tallahassee, Florida, USA
import os
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
plt.rcParams['text.usetex'] = True

import tree
#import numpy.ma as ma

def pick_two_lineages(population,pop,kpop):
    #x = ma.array(population, mask=ma.masked_not_equal(population, pop))  
    x = [i for i in range(len(population)) if population[i].pop == pop]
    y = [i for i in range(len(population)) ]
    #print("x:", [population[xi].pop for xi in x],len(population),kpop) 
    #print("y:", [population[xi].pop for xi in y],len(population) )  
    i,j = np.random.choice(x, size=2, replace=False)
    return i,j

def pick_one_lineage(population, pop):
    x = [i for i in range(len(population)) if population[i].pop == pop]
    i = np.random.choice(x, size=1, replace=False)
    return i

def setup_population(samples):
    numberpopulations = len(samples)
    #print(samples)
    populations = []
    for pop in range(numberpopulations):
        for i in range(samples[pop]):
            xx = tree.Node(pop)
            xx.name = str(pop) + '_' + str(i) 
            populations.append(xx)
    return populations   

def connect_lineages(population, i, j, age ):
    #mypopulation = population[:]
    #print("population:", population)
    #print("ind:", population[i].pop, population[j].pop)
    q  = tree.Node()
    q.pop = population[i].pop 
    q.age = age
    population[i].ancestor = q
    population[j].ancestor = q
    q.left = population[i]
    q.right = population[j]
    population[i] = q 
    population.pop(j)
    return population

@jit(nopython=True)
def mylambda(theta,M,k,alphas):    
    npop = len(theta)
    npop2 = npop**2
    #if alpha==None:
    #    alphas = np.ones(npop)
    #else:
    #    alphas = np.array(alpha,dtype=np.float64)     
    l = fill_mygammas(alphas)
    print("gammas:",np.exp(l))
    xlambdas = np.zeros(npop2)
    for i in range(npop):
        if k[i]> 1:
            xlambdas[i] =   k[i] * ( k[i]-1 ) / theta[i]
        else:
            xlambdas[i] = 0.0
    z = npop
    for i in range(npop):
        for j in range(npop): 
            if i != j:
                xlambdas[z] = M[j]*k[i]
                z += 1  
    Y = xlambdas*np.exp(l)
    print("k:",k)
    print("xlambdas:",xlambdas)
    print("Y:",Y)
    return Y,xlambdas,l



@jit(nopython=True)
def lgamma(x):
    return np.log(np.math.gamma(x))

@jit(nopython=True)
def fill_mygammas(alphas):
    npop = len(alphas)
    npop2 = npop**2
    mygammas = np.zeros(npop2)
    allgamma=0
    for i in range(npop):
        allgamma += lgamma(alphas[i]+1)
    z = npop
    for i in range(npop):
        # missing a term? mygammas[i] = allgamma - lgamma (alphas[i]+1)
        mygammas[i] = - lgamma (alphas[i]+1) #allgamma - lgamma() - allgamma
        #print(f"mygammas[{i}]={mygammas[i]}")
        for j in range(npop): 
            if i != j:
                #mygammas[z] = (allgamma - lgamma(alphas[i]+1)) - allgamma
                mygammas[z] = - lgamma(alphas[i]+1)
                #print(f"mygammas[{z}]={mygammas[z]}")
                z += 1  
    return mygammas

@jit(nopython=True)
def fill_Yalphas(alphas):
    npop = len(alphas)
    npop2 = npop**2
    Yalphas = np.zeros(npop2)
    z = npop
    for i in range(npop):
        Yalphas[i] = alphas[i]
        for j in range(npop): 
            if i != j:
                Yalphas[z] = alphas[i]
                z += 1  
    return Yalphas


#Somayeh Mashayekhi                                                           
#double propose_new_mlftime(double lambda, double alpha, double r1, double r2)
#{
#  double pia = PI * alpha;
#  double denoma = 1.0 / alpha;
#  double denomlambda = 1.0 / lambda;
#  return -pow(denomlambda,denoma) * pow((sin(pia)/(tan(pia*(1.0-r1))) - cos(pia\
#)),denoma) * log(r2);
#  //return -pow(denomlambda * (sin(pia)/(tan(pia*(1-r1))) - cos(pia)),denoma) *\
# log(r2);                                                                       
#}

@jit(nopython=True)
def randommlftime(mylambda, alpha):
    pia = 3.1415926 * alpha
    r1 = np.random.uniform(0,1)
    r2 = np.random.uniform(0,1)
    denoma = 1.0 / alpha
    if mylambda == 0.0:
        return 1e100
    denomlambda = 1.0 / mylambda
    return -denomlambda**denoma * (np.sin(pia)/(np.tan(pia*(1.0-r1)))-np.cos(pia))**denoma * np.log(r2)

@jit(nopython=True)
def randomtime(Y,alphas,t0):
    smallu = 1e100
    for yi,ai in zip(Y,alphas):
        u = randommlftime(yi,ai)
        if u < smallu:
            smallu = u
    return t0 + smallu

#@jit(nopython=True)
def kingmanrandomtime(Y,t0):
    #lY = len(Y)
    Y = np.sum(Y)
    #Y = Y + 1e-10
    
    #u =  (np.log(Y) - np.log(np.random.uniform(0,1,lY)))/Y 
    #u =  ( - np.log(np.random.uniform(0,1,lY)))/Y
    u =  ( - np.log(np.random.uniform(0,1)))/Y 
    #u[u<=0] = 1000000
    return t0 + u

#@jit(nopython=True)
def oldrandomtime(Y,t0):
    lY = len(Y)
    Y = Y + 1e-10
    #u =  (np.log(Y) - np.log(np.random.uniform(0,1,lY)))/Y 
    u =  ( - np.log(np.random.uniform(0,1,lY)))/Y 
    u[u<=0] = 1000000
    return t0 + u

@jit(nopython=True)
def mm2m(i,numpop):
    #print(i)
    if i<numpop:
        return i, i
    else:
        topop = (i - numpop) // (numpop - 1)
        frompop = i - numpop - topop * (numpop - 1)
        if frompop >= topop:
            frompop += 1;
        return int(frompop),int(topop)

def pick_event(lambdas):
    s = np.sum(lambdas)
    x = lambdas/s
    print(x)
    return np.random.choice(range(len(lambdas)), p=x)

    
import sys 
#@jit(nopython=True)
def sim_one(ne,M,savek,alpha,numpop):
    
    populations = setup_population(savek)
    
    k = savek.copy()
    count = 0
    otmin=0.0
    mytimes=np.zeros(100000)
    while np.sum(k)>1 and count < 100000:
        count += 1
        Y,debuglambdas,debugl = mylambda(ne,M,k,alpha)
        Yalphas = fill_Yalphas(alpha)
        if DEBUG:
            print(k, Y, debuglambdas,np.exp(debugl),file=sys.stderr)
        t1 = randomtime(Y,Yalphas, otmin)
        #print("time choices:", t1,end=' | ')
        #mini = np.argmin(t1)
        #tmin = t1[mini]
        tmin = t1
        u = tmin - otmin
        mytimes[count] = u
        otmin = tmin
        mini=pick_event(debuglambdas)
        print("index:",mini, "time:",tmin, "interval:",u)
        frompop, topop = mm2m(mini,numpop)
        if frompop == topop:
            #print(populations[frompop])
            if k[frompop] < 2:
                #print("k[frompop] < 2")
                break
            l1,l2 = pick_two_lineages(populations,frompop,k[frompop])
            #print("pick:",frompop,"::", l1,l2)
            populations = connect_lineages(populations,l1,l2, tmin)
            k[frompop] -= 1
        else:
            l1 = pick_one_lineage(populations,topop)[0]
            #print("pick2:",topop,"::", l1)
            populations[l1].pop = frompop              
            k[topop] -= 1
            k[frompop] += 1   
        #print(f"{tmin} {frompop} :: {k}  {[i.pop for i in populations]}")
    return populations  

#@jit(nopython=True)
def run_sim(nrun, ne,M,k,alpha,alpha2):
    numpop = len(ne)
    simtimes=np.zeros(nrun)
    simtimes2=np.zeros(nrun)
    for i in np.arange(0,nrun): 
        simtimes[i] = np.sum(sim_one(ne,M,k,alpha,numpop))
        simtimes2[i] = np.sum(sim_one(ne,M,k,alpha2,numpop))
    return simtimes,simtimes2

def report_stats(simtimes, loci, title=''):
    print(title)
    print("TMRCA         = ",np.mean(simtimes))
    print("TMRCA:std     = ",np.std(simtimes))
    print("TMRCA:min,max = ",np.min(simtimes),np.max(simtimes))   
    print("Loci          = ",loci)


import argparse as ap

if __name__ == '__main__':
    DEBUG = False
    parser = ap.ArgumentParser(description='Simulate a tree')
    parser.add_argument('-l', '--loci', type=int, default=1, help='number of loci')
    parser.add_argument('-s', '--sites', type=int, default=1000, help='number of sites')
    parser.add_argument('-i', '--individuals', type=str, default="10,10", help='Number of samples for each population')    
    parser.add_argument('-t', '--theta', type=str, default="0.01,0.01", help='thetas for each population')    
    parser.add_argument('-m', '--mig', type=str, default="100,100", help='migration rate for each population')
    parser.add_argument('-a', '--alpha', type=str, default="1.0,1.0", help='alpha for each population')
    parser.add_argument('-f', '--file', type=str, default="NONE", help='treefile to be used with migdata, default is NONE which is a placeholder for sys.stdout')
    parser.add_argument('-p', '--plot', action='store_true', help='Plots density histogram of TMRCA')
    args = parser.parse_args()
    myfile = args.file
    sites = args.sites
    loci = args.loci    
    plot = args.plot
    k=[int(ki) for ki in args.individuals.split(',')]  
    #[10,10]
    ne = [float(ki) for ki in args.theta.split(',')]   
    #np.array([0.01,0.01]) 
    M = [float(ki) for ki in args.mig.split(',')]       
    #np.array([100.,100.])
    alpha = [float(ki) for ki in args.alpha.split(',')]   
    #np.array([0.9,0.9])
    alpha2 = np.array([1.0,1.0])
    if myfile == 'NONE':
        thetree = sys.stdout 
    else:
        thetree = open(myfile,'w')
    nl=os.linesep
    simtimes=[] 
    simtimes2=[] 
    r = np.random.randint(1000000)
    thetree.write(f"#SN{nl}#{r}{nl}#2{nl}#{k[0]} {k[1]}{nl}#{loci} {sites} 2.0{nl}")
    for locus in range(loci):
        thetree.write(f"# rate among sites for locus 0 (1.000000){nl}#={nl}")

    for locus in range(loci):
        thetree.write(f"#$ locus {locus}{nl}#$ 0.000001{nl}")
        root = sim_one(ne,M,k,alpha,2)
        if plot:
            root2 = sim_one(ne,M,k,alpha2,2)
            simtimes2.append((root2[0].age))
        #print(root)
        t = tree.Tree()
        t.root = root[0]
        simtimes.append((root[0].age))
        t.myprint(root[0],file=thetree)
        thetree.write(f":0.0;{nl}")

    report_stats(simtimes,loci,f'{alpha}')            
    if myfile != 'NONE':
        thetree.close()

    #simtimes,simtimes2 = run_sim(1000, ne,M,k,alpha,alpha2)
    if plot:
        report_stats(simtimes2,loci,f'{alpha2}')  
        plt.hist(simtimes, bins=50,color='blue', histtype='step', density=True, alpha=0.6, label=r'$\alpha=$'+f'{alpha}')
        plt.hist(simtimes2, bins=50, density=True, alpha=0.6, color='red',histtype='step', label=r'$\alpha=1.0$')
        plt.legend(loc='upper right')
        plt.xlim(0,0.12)
        plt.ylim(0,80.0)
        plt.xlabel('Time to MRCA')
        plt.ylabel('Density')
        plt.savefig('simtreefig.pdf')
    
    
