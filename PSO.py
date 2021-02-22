# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:30:25 2019

@author: Faisal Caem
"""
import numpy as np
population= 200
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
generation=250
pCrazy=0.1
Xub=10 #x upper bound
Xlb=8 #x lower bound
Yub=13 #y upper bound
Ylb=10
#%%
#initializing swarm
def swarm():
    popul=[]
    while len(popul)<population:
        poss=[]
        x=np.random.uniform(Xlb,Xub)
        y=np.random.uniform(Ylb,Yub)
        if x+y<=22:
            poss+=[x,y]
            popul.append(poss)
    return popul
#%%
#evaluation fitness of swarm
def evaluation(x,y):
    Fx=-x*math.sin(4*x)-1.1*y*math.sin(2*y)+2
    return Fx
#%%
#updating pBest
def pBest(lastPop, currentPop):
    pop=[]
    for i in range(len(lastPop)):
        eval_lastPop=evaluation(lastPop[i][0],lastPop[i][1])
        eval_currentPop=evaluation(currentPop[i][0],currentPop[i][0])
        if eval_currentPop>eval_lastPop:
            pop.append(currentPop[i])
        else:
            pop.append(lastPop[i])
    return pop
#%%
#updating gBest
def gBest(pop):
    fitness=[]
    for i in range (len(pop)):
        evalue=evaluation(pop[i][0],pop[i][1])
        fitness.append(evalue)
    bestPop=pop[np.argmax(fitness)]
    return bestPop
#%%
def updateVelocity(velocity, initialPop, currentPop):
    globalBest=gBest(initialPop)
    personalBest=pBest(initialPop, currentPop)
    c1=c2=2.0775
    phi=c1+c2
    r1=np.random.uniform(0,1)
    r2=np.random.uniform(0,1)
    k=2/abs(2-phi-math.sqrt(phi**2-4*phi))
    nowVelocity=[]
    for i in range (len(velocity)):
        nowVeloc=[]
        for j in range(len(velocity[i])):
            currentVelocity=k*(velocity[i][j]+r1*c1*(personalBest[i][j]-initialPop[i][j])+r2*c2*(globalBest[j]-initialPop[i][j]))
            nowVeloc.append(currentVelocity)
        nowVelocity.append(nowVeloc)
    return nowVelocity
#%%
#particle craziness
def crazinessParticle(individu):
    rand=np.random.uniform(0,1)
    k=np.random.uniform(-1,1)
    Vx=k*2
    Vy=k*3
    V=[rand*Vx,rand*Vy]
    for i in range (len (individu)):
        individu[i]+=V[i]
    return individu
#%%
def euclidian_distance(coordinate1, coordinate2):
    dist=math.sqrt((coordinate2[1]-coordinate1[1])**2 + (coordinate2[0]-coordinate1[0])**2)
    return dist

#%%
def main():
    initialPop=swarm()
    velocity=[]
    for i in range (len(initialPop)):
        Vx=np.random.uniform(-1,1)
        Vy=np.random.uniform(-1,1)
        V=[Vx,Vy]
        velocity.append(V)
    nextVelocity=deepcopy(velocity)
    currentPop=deepcopy(initialPop)
    iteration=0
    maxFitness=[]
    averageFitness=[]
    while iteration<generation:
        #print('Iteration {}'.format(iteration+1))
        iteration+=1
        nextPopulation=[]
        for i in range(len(currentPop)):
            nowPop1=currentPop[i][0]+nextVelocity[i][0]
            nowPop2=currentPop[i][1]+nextVelocity[i][1]
            """if nowPop1 < Xlb:
                cosA= (math.sqrt((nowPop1-currentPop[i][0])**2)/
                       math.sqrt((nowPop1-currentPop[i][0])**2 + (nowPop2-currentPop[i][1])**2))
                if nowPop2-currentPop[i][1]>0:
                    nowPop1=Xlb
                    nowPop2=currentPop[i][1] + (math.sqrt((Xlb-currentPop[i][0])**2)/cosA)
                elif nowPop2-currentPop[i][1]<0:
                    nowPop1=Xlb
                    nowPop2=currentPop[i][1] - (math.sqrt((Xlb-currentPop[i][0])**2)/cosA)
            elif nowPop1 >Xub:
                cosA= (math.sqrt((nowPop1-currentPop[i][0])**2)/
                       math.sqrt((nowPop1-currentPop[i][0])**2 + (nowPop2-currentPop[i][1])**2))
                if cosA==0:
                    nowPop2= currentPop[i][1]
                    nowPop1=Xub
                elif nowPop2-currentPop[i][1]>0:
                    nowPop1=Xub
                    nowPop2=currentPop[i][1] + (math.sqrt((Xub-currentPop[i][0])**2)/cosA)
                elif nowPop2-currentPop[i][1]<0:
                    nowPop1=Xub
                    nowPop2=currentPop[i][1] - (math.sqrt((Xub-currentPop[i][0])**2)/cosA)
            elif nowPop2 < Ylb:
                cosA= (math.sqrt((nowPop2-currentPop[i][1])**2)/
                       math.sqrt((nowPop1-currentPop[i][0])**2 + (nowPop2-currentPop[i][1])**2))
                if cosA==0:
                    nowPop2=Ylb
                    nowPop1=currentPop[i][0]
                elif nowPop1-currentPop[i][0] >0:
                    nowPop2=Ylb
                    nowPop1=currentPop[i][0] + (math.sqrt((Ylb-currentPop[i][1])**2)/cosA)
                elif nowPop1-currentPop[i][0]<0:
                    nowPop2=Ylb
                    nowPop1=currentPop[i][0] - (math.sqrt((Ylb-currentPop[i][1])**2)/cosA)
            elif nowPop2 > Yub:
                cosA= (math.sqrt((nowPop2-currentPop[i][1])**2)/
                       math.sqrt((nowPop1-currentPop[i][0])**2 + (nowPop2-currentPop[i][1])**2))
                if cosA==0:
                    nowPop2=Yub
                    nowPop1=currentPop[i][0]
                elif nowPop1-currentPop[i][0] >0:
                    nowPop2=Yub
                    nowPop1=currentPop[i][0] + (math.sqrt((Yub-currentPop[i][1])**2)/cosA)
                elif nowPop1-currentPop[i][0]<0:
                    nowPop2=Yub
                    nowPop1=currentPop[i][0] - (math.sqrt((Yub-currentPop[i][1])**2)/cosA)
            nowPop=[nowPop1,nowPop2]
            rand=np.random.uniform(0,1)
            if rand<=pCrazy:   
                nowPopulation=crazinessParticle(nowPop)
                if nowPopulation[0] < Xlb:
                    cosA= (math.sqrt((nowPopulation[0]-nowPop[0])**2)/
                           math.sqrt((nowPopulation[0]-nowPop[0])**2 + (nowPopulation[1]-nowPop[1])**2))
                    if nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[0]=Xlb
                        nowPopulation[1]=nowPop[1] + (math.sqrt((Xlb-nowPop[0])**2)/cosA)
                    elif nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[0]=Xlb
                        nowPopulation[1]=nowPop[1] - (math.sqrt((Xlb-nowPop[0])**2)/cosA)
                elif nowPopulation[0] > Xub:
                    cosA= (math.sqrt((nowPopulation[0]-nowPop[0])**2)/
                           math.sqrt((nowPopulation[0]-nowPop[0])**2 + (nowPopulation[1]-nowPop[1])**2))
                    if nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[0]=Xub
                        nowPopulation[1]=nowPop[1] + (math.sqrt((Xub-nowPop[0])**2)/cosA)
                    elif nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[0]=Xub
                        nowPopulation[1]=nowPop[1] - (math.sqrt((Xub-nowPop[0])**2)/cosA)
                elif nowPopulation[1] < Ylb:
                    cosA= (math.sqrt((nowPopulation[1]-nowPop[1])**2)/
                           math.sqrt((nowPopulation[0]-nowPop[0])**2 + (nowPopulation[1]-nowPop[1])**2))
                    if cosA==0:
                        nowPopulation[0]= nowPop[0]
                        nowPopulation[1]=Ylb
                    elif nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[1]=Ylb
                        nowPopulation[0]=nowPop[0] + (math.sqrt((Ylb-nowPop[0])**2)/cosA)
                    elif nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[1]=Ylb
                        nowPopulation[0]=nowPop[0] - (math.sqrt((Ylb-nowPop[0])**2)/cosA)
                elif nowPopulation[1] > Yub:
                    cosA= (math.sqrt((nowPopulation[1]-nowPop[1])**2)/
                           math.sqrt((nowPopulation[0]-nowPop[0])**2 + (nowPopulation[1]-nowPop[1])**2))
                    if cosA==0:
                        nowPopulation[0]= nowPop[0]
                        nowPopulation[1]=Yub
                    if nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[1]=Yub
                        nowPopulation[0]=nowPop[0] + (math.sqrt((Yub-nowPop[0])**2)/cosA)
                    elif nowPopulation[1]-nowPop[1]>0:
                        nowPopulation[1]=Yub
                        nowPopulation[0]=nowPop[0] - (math.sqrt((Yub-nowPop[0])**2)/cosA)
            else:
                nowPopulation=nowPop"""
            if nowPop1<8:
                nowPop1=8
            elif nowPop1>10:
                nowPop1=10
            if nowPop2<10:
                nowPop2=10
            elif nowPop2>13:
                nowPop2=13
            nowPop=[nowPop1,nowPop2]
            rand=np.random.uniform(0,1)
            if rand<=pCrazy:   
                nowPopulation=crazinessParticle(nowPop)
                if nowPopulation[0]<8:
                    nowPopulation[0]=8
                elif nowPopulation[0]>10:
                    nowPopulation[0]=10
                if nowPopulation[1]<10:
                    nowPopulation[1]=10
                elif nowPopulation[1]>13:
                    nowPopulation[1]=13
            else:
                nowPopulation=nowPop
            nowPopulation=[nowPop1,nowPop2]
            nextPopulation.append(nowPopulation)
        arr_individuBest=pBest(currentPop, nextPopulation)
        evaluate=[]
        for i in range(len(arr_individuBest)):
            evalue=evaluation(arr_individuBest[i][0], arr_individuBest[i][1]) #x and y kalo mau nyari
            evaluate.append(evalue)
        nextVeloc=updateVelocity(nextVelocity, currentPop,arr_individuBest)
        nextVelocity=deepcopy(nextVeloc)
        currentPop=deepcopy(arr_individuBest)
        maxFit=max(evaluate)
        averageFit=np.mean(evaluate)
        maxFitness.append(maxFit)
        averageFitness.append(averageFit)
    #print('-----------Final Value is {}-----------'.format(maxFit))
    return nextPopulation, maxFitness, averageFitness
#%%
import time
time_start = time.time()
i=0
mmax=[]
print('iterating')
while i <30:
    i+=1
    PSO=main()
    mmax.append(max(PSO[1]))
print('finish')
fig = plt.figure(figsize = (8, 5))
plt.subplots_adjust(hspace = 0.5)
plt.plot(PSO[1], label = 'f(x,y) max')
plt.xlabel('Generation', fontsize = 15)
plt.ylabel('f(x,y)', fontsize = 15)
plt.grid()
plt.plot(PSO[2], label = 'f(x,y) average')
plt.legend(fontsize= 10, loc = 4)

time_elapsed = (time.time() - time_start)
print("Total Time is " + str(time_elapsed) +'s')
#%%
df=pd.DataFrame(mmax)
filepath=r'G:\NTUST\Lab Data Mining\K means\pso.xlsx'
df.to_excel (filepath, index=False)