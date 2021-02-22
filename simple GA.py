# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:47:22 2019
Simple GA
@author: Faisal Fuad Nursyahid
"""
population=30 #predefined
chromosomeX=11
chromosomeY=12
chromosome=chromosomeX + chromosomeY
generation=100 #predefined
crossover_rate=0.85 #sumber slide
mutation_rate=0.1 #sumber slide
Xub=10 #x upper bound
Xlb=8 #x lower bound
Yub=13 #y upper bound
Ylb=10
import pandas as pd
import numpy as np
import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt
#%%
#initializing population
def initialize():
    popul=[]
    for i in range (population):
        person=[]
        for j in range (chromosome):
            individu=random.randint(0,1)
            person.append(individu)
        popul.append(person)
    return popul
#%%
#evaluating fitness function for every chromosome
def evaluation(x,y):
    Fx=-x*math.sin(4*x)-1.1*y*math.sin(2*y)+2
    return Fx
#%%
#converting binary chromosome into a decimal number
def linearization(binary):
    Xbin=binary[:chromosomeX:-1]
    Ybin=binary[chromosomeX::-1]
    Xdec=0
    Ydec=0
    XBinub=0
    YBinub=0
    for i in range (len(Xbin)):
        Xdec+=Xbin[i]*2**i
        XBinub+=2**i
    for j in range (len (Ybin)):
        Ydec+=Ybin[j]*2**j
        YBinub+=2**j
    Xlin=Xlb+((Xub-Xlb)/(XBinub-0)*Xdec)
    Ylin=Ylb+((Yub-Ylb)/(YBinub-0)*Ydec)
    return Xlin,Ylin
#%%
#Two point reverse swap crossover
'''def crossover(father, mother):
    [rand1,rand2]=sorted(random.sample(range(0,chromosome),2))
    swap1=father[rand1+1:rand2]
    swap2=mother[rand1+1:rand2]
    swap1.reverse()
    swap2.reverse()
    child1=father[:rand1+1]+swap2+father[rand2:]
    child2=mother[:rand1+1]+swap1+mother[rand2:]
    return child1, child2'''
#%%
#uniform crossover
def crossover(father, mother):
    #uniform crossover
    mask=[]
    for i in range (chromosome):
        rand=random.randint(0,1)
        mask.append(rand)
    offspring1=[]
    offspring2=[]
    for j in range (len (mask)):
        if mask[j]==0:
            offspring1.append(father[j])
            offspring2.append(mother[j])
        else:
            offspring1.append(mother[j])
            offspring2.append(father[j])
    return offspring1, offspring2
#%%
#one bit mutation
def mutation(individu):
    [rand]=random.sample(range(0,chromosome),1) #index
    if individu[rand]==1:
        individu[rand]=0
    else:
        individu[rand]=1
    return individu
#%%
#bit wise mutation
'''def mutation(individu):
    for i in range (len (individu)):
        rand=np.random.uniform()
        if rand<=mutation_rate:
            if individu[i]==1:
                individu[i]=0
            else:
                individu[i]=1
        else:
            individu[i]=individu[i]
    return individu'''
#%%
#Roulette wheel selection
def selection(pop):
    fitness=[]
    popul=pd.DataFrame(pop)
    for i in range (len (pop)):
        linear=linearization(pop[i])
        if linear[0]+linear[1]<=22: #constraint
            fit=evaluation(linear[0],linear[1])
            fitness.append(fit)
        else:
            penalty=abs((22-linear[0]-linear[1])/2) #repairing infeasible chromosome
            fit=evaluation((linear[0]-penalty),(linear[1]-penalty))
            fitness.append(fit)
    popul['f(x,y)']=[float(i) for i in fitness]
    absolute=abs(min(popul['f(x,y)']))
    total=0
    for j in range (len (fitness)):
        total+=absolute+popul['f(x,y)'][j]
    probability=[]
    for k in range(len (fitness)):
        probability.append(abs(absolute+fitness[k])/total)
    popul['Probability']=[float(i) for i in probability]
    cumm=0
    cummulative=[]
    for l in range (len (fitness)):
        cumm+=popul['Probability'][l]
        cummulative.append(cumm)
    popul['Cummulative']=[float(i) for i in cummulative]
    select=[]
    [rand1,rand2]=np.random.uniform(0,1,2)
    for i in range (len (popul['Cummulative'])):
        if rand1 <= popul['Cummulative'][i]:
            choosen1=list(map(int, popul.iloc[i][:-3].values.tolist()))
            break
    select.append(choosen1)
    for i in range (len (popul['Cummulative'])):
        if rand2 <= popul['Cummulative'][i]:
            choosen2=list(map(int, popul.iloc[i][:-3].values.tolist()))
            break
    select.append(choosen2)
    return select
    
#%% 
#main program
def main():
    initialPopulation=initialize() #initialize chromosome (matrix dim: pop x chromosome)
    gene=0
    nextPopulation=deepcopy(initialPopulation)
    maxFitness=[]
    averageFitness=[]
    while gene<generation:
        print('Iteration {}'.format(gene+1))
        gene+=1
        popul=0
        newPopul=[]
        while popul<(population/2):
            selectParent=selection(nextPopulation)
            rand1=np.random.uniform(0,1) #random uniform
            if rand1 <crossover_rate: #crossover
                children=crossover(selectParent[0], selectParent[1])
                popul+=1
                newChildren=[]
                for i in range (len (children)):
                    rand2=np.random.uniform(0,1)
                    if rand2<1-mutation_rate:#mutation
                        newChildren.append(children[i])
                    else:
                        newChild=mutation(children[i])
                        newChildren.append(newChild)
                for j in range (len (newChildren)):
                    newPopul.append(newChildren[j])
            else:
                popul=popul
        evalue=[]
        for i in range (len (nextPopulation)):
            value=linearization(nextPopulation[i])
            evaluate=evaluation(value[0],value[1])
            evalue.append(evaluate)
        idx=np.argmax(evalue)
        newPopul.append(nextPopulation[idx])
        evalue=[]
        for j in range (len (newPopul)):
            value=linearization(newPopul[j])
            evaluate=evaluation(value[0],value[1])
            evalue.append(evaluate)
        idx=np.argmin(evalue)
        del(newPopul[idx])
        nextPopulation=deepcopy(newPopul)
        evalue=[]
        for i in range (len (nextPopulation)):
            value=linearization(nextPopulation[i])
            evaluate=evaluation(value[0],value[1])
            evalue.append(evaluate)
        maxFit=max(evalue)
        averageFit=np.mean(evalue)
        maxFitness.append(maxFit)
        averageFitness.append(averageFit)
    print('-----------Final Value is {}-----------'.format(maxFit))
    return maxFitness,averageFitness 
#%%
import time
time_start = time.clock()
i=0
mmax=[]
while i <30:
    i+=1
    GA=main()
    mmax.append(max(GA[0]))
fig = plt.figure(figsize = (8, 5))
plt.subplots_adjust(hspace = 0.5)
plt.plot(GA[0], label = 'f(x,y) max')
plt.xlabel('Generation', fontsize = 15)
plt.ylabel('f(x,y)', fontsize = 15)
plt.grid()
plt.plot(GA[1], label = 'f(x,y) average')
plt.legend(fontsize= 10, loc = 4)

time_elapsed = (time.clock() - time_start)
print("Time is " + str(time_elapsed) +'s')
#%%
df=pd.DataFrame(mmax)
filepath=r'G:\NTUST\Lab Data Mining\K means\GA.xlsx'
df.to_excel (filepath, index=False)