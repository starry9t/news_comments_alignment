#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:15:20 2019

@author: Yu Zhou
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:43:09 2019

@author: Yu Zhou
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(r"/home/zhouyu/github/acvae/src")
import subprocess
import numpy as np
#from sklearn.datasets import fetch_20newsgroups
import argparse
from random import sample
import itertools
import time
import pickle
import math

from progressbar import Bar, ETA, FileTransferSpeed, Percentage, ProgressBar
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from palmettopy.palmetto import Palmetto
from tensorboardX import SummaryWriter
from models import standardVAE, OIVAE, DISCRIMINATOR, OIVAESOCC
from training import zeroOne, writeTWC, getAtopic, getTopicWords, buildCorpusDict, evaluate, evaluateSOCC, writePRF, writeYahooAnswer, writeSOCCAnswer, writeSOCCAnswers
from mypublic import pk, loadpk, batchPair, compareTopicChanging, writeTopics, writeTensor, predLabel, crossValidation, loadGlobals, oldbuildDataMatrix, oldbuildDicts



##########################20 news group #####################

#data_dir = os.path.join(cur_dir, 'data/ng')
#output_dir = os.path.join(cur_dir, 'output/ng')
#txtFile = os.path.join(data_dir, 'train.txt')
#labelFile = os.path.join(data_dir, 'label.txt')
#
#def to_onehot(data, min_length):
#    return np.bincount(data, minlength=min_length)
#
#def makeData(file):
#    global iwDict, wiDict, X, Y, dimV
#    finalDict, wiDict, iwDict = oldbuildDicts(file, 999999, 0)
#    X = oldbuildDataMatrix(file, wiDict)
#    X = torch.from_numpy(X.astype(np.float32))
#    X = Variable(X)
#    Y = torch.randn(X.size(0))
#    dimV = len(wiDict)
#    print('Training Data Size: {}'.format(X.size()))

#########################20 news group ######################
        
def makeOptimizer(modelVAE, modelOIVAE, discriminator):
    global optimizerVAE, optimizerOIVAE, optimizerD
    if args.optimizer == 'Adam':
        optimizerVAE = torch.optim.Adam(modelVAE.parameters(), args.lr, betas=(args.mmt, 0.999))
        optimizerOIVAE = torch.optim.Adam(modelOIVAE.parameters(), args.lr, betas=(args.mmt, 0.999))
        optimizerD = torch.optim.Adam(discriminator.parameters(), args.lr, betas=(args.mmt, 0.999))
    elif args.optimizer == 'SGD':
        optimizerVAE = torch.optim.SGD(modelVAE.parameters(), args.lr, momentum=args.mmt)
        optimizerOIVAE = torch.optim.SGD(modelOIVAE.parameters(), args.lr, momentum=args.mmt)
        optimizerD = torch.optim.SGD(discriminator.parameters(), args.lr, momentum=args.mmt)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

def one_hot(idx, size, cuda=False):
    a = np.zeros((1, size), np.float32)
    for i in idx:
        a[0][i] += 1
    v = Variable(torch.from_numpy(a))
    if args.gpu: v = v.cuda()
    return v
    
def pair2torch(pairs):
    t = torch.tensor([])
    if args.gpu:
        t = t.cuda()
    for (aID, cID) in pairs:
        aTensor = one_hot(articleId2Idx[aID], dimV)
        cTensor = one_hot(commId2Idx[cID], dimV)
        xTensor = aTensor+cTensor
        t = torch.cat((t,xTensor), dim=0)
    t = Variable(t)
    return t

def label2torch(labels):
    t = torch.LongTensor(labels).type(torch.LongTensor) 
    if args.gpu:
        t = t.cuda()
    t = Variable(t)
    return t
    
def splitData(pairs, tag='a'):
    t = torch.tensor([])
    if args.gpu:
        t = t.cuda()
    if tag == 'a':
        for (aID, cID) in pairs:
            aTensor = one_hot(articleId2Idx[aID], dimV)
            t = torch.cat((t,aTensor), dim=0)
    elif tag == 'c':
        for (aID, cID) in pairs:
            cTensor = one_hot(articleId2Idx[aID], dimV)
            t = torch.cat((t,cTensor), dim=0)   
    else:
        raise TypeError
    t = Variable(t)
    return t  

def train(args, modelVAE, modelOIVAE, discriminator, X, Y, iwD, numTopic):   
    widgets = ['T{}Running: '.format(numTopic), Percentage(), ' ', Bar(), ' ', ETA()]
    batchNum = math.ceil(len(X)/args.batchSize)
    pbar = ProgressBar(widgets=widgets, maxval=(args.numEpoch*batchNum*3))
    pbar.start()

    LossWriter = SummaryWriter(os.path.join(t_dir,'runs/loss'))
    KLWriter = SummaryWriter(os.path.join(t_dir,'runs/KLD'))
    RecWriter = SummaryWriter(os.path.join(t_dir,'runs/Rec'))
 
    AtopicWords4Epochs = []
    CtopicWords4Epochs = []

    modelOIVAE.train(True)
    #adversarial_loss = torch.nn.BCELoss()
    for epoch in range(args.numEpoch):
        eloss = 0.0; ekll=0.0; erel=0.0
        allIndices = torch.randperm(len(X)).split(args.batchSize)
        
        # train model2
        for bi,batchIndices in enumerate(allIndices):
            bPair = batchPair(X, batchIndices)
            dataOIVAE = pair2torch(bPair)    
            ybPair = batchPair(Y, batchIndices)
            ydataOIVAE = label2torch(ybPair)
            #dataNG = Variable(X[batchIndices]) # 20 newsgroup
            optimizerOIVAE.zero_grad()
            y2, kll2, rel2, loss2 = modelOIVAE(dataOIVAE, ydataOIVAE, compute_loss=True, avg_loss=True) 
            oivaeZa = modelOIVAE.decoder.getZa(modelOIVAE.getP(dataOIVAE))
            resD2, lossD2 = discriminator(oivaeZa, ydataOIVAE, compute_loss=True)
            oivaeLoss = loss2+lossD2
            oivaeLoss.backward()
            optimizerOIVAE.step()   
            eloss += loss2
            ekll += kll2
            erel += rel2
            pbar.update(((epoch*batchNum*3)+(bi+1)))
             
        #train model1
        for bi,batchIndices in enumerate(allIndices):
            bPair = batchPair(X, batchIndices)
            dataVAE = splitData(bPair, tag='a') 
            ybPair = batchPair(Y, batchIndices)
            ydataVAE = label2torch(ybPair)
            #dataNG = Variable(X[batchIndices]) # 20 newsgroup
            optimizerVAE.zero_grad()
            y, kll, rel, loss = modelVAE(dataVAE, ydataVAE, compute_loss=True, avg_loss=True) 
            vaeZa = modelVAE.getP(dataVAE, compute_loss=False)
            resD, lossD = discriminator(vaeZa, ydataVAE, compute_loss=True)
            vaeLoss = loss+lossD
            vaeLoss.backward()
            optimizerVAE.step()
            pbar.update(((epoch*batchNum*3)+batchNum+(bi+1)))
        
        # train discriminator
        for bi,batchIndices in enumerate(allIndices):
            bPair = batchPair(X, batchIndices)
            fakeData = pair2torch(bPair)
            realData = splitData(bPair)
            fakeLabels = torch.LongTensor([0 for i in range(len(batchIndices))])
            realLabels = torch.LongTensor([1 for i in range(len(batchIndices))])
            realz = modelVAE.getP(realData, compute_loss=False)
            fakez = modelOIVAE.decoder.getZa(modelOIVAE.getP(fakeData))
            if args.gpu:
                realz = realz.cuda() ; fakez = fakez.cuda() 
                fakeLabels = fakeLabels.cuda() ; realLabels = realLabels.cuda()
            optimizerD.zero_grad()
            resR, lossR = discriminator(realz, realLabels, compute_loss=True)
            resF, lossF = discriminator(fakez, fakeLabels, compute_loss=True)
            d_loss = (lossR + lossF) / 2
            d_loss.backward()
            optimizerD.step()
            pbar.update(((epoch*batchNum*3)+batchNum*2+(bi+1)))
            
        LossWriter.add_scalar('loss', eloss, epoch)
        KLWriter.add_scalar('KLD', ekll, epoch)
        RecWriter.add_scalar('recloss', erel, epoch)
        
        if (epoch)%10 == 0:
            # down-stream task
            if args.task == 'a':
                p,r,f = testAlign(modelOIVAE)
                print('p: {}, r:{}, f:{}'.format(p,r,f))
                # at this epoch
                # numTopic = 
                prfFile = os.path.join(e_dir, 'resultLog.txt')
                writePRF(prfFile,p,r,f,epoch,numTopic)
                if f > 0.3:                 
                    # record topic word at this point
                    atwtensor = modelOIVAE.decoder.fcG1.weight.data
                    aTopicWords = getAtopic(atwtensor,10)
                    aTopicWords = getTopicWords(aTopicWords, iwD)
                    AtopicWords4Epochs.append(aTopicWords) 
                    ctwtensor = modelOIVAE.decoder.fcG2.weight.data
                    cTopicWords = getAtopic(ctwtensor,10)
                    cTopicWords = getTopicWords(cTopicWords, iwD)
                    CtopicWords4Epochs.append(cTopicWords)
                if f > 0.5:
                    fn = os.path.join(t_dir, 'soccResult.txt')
                    writeSOCCAnswers(predictedLabel, testLabels, fn, Idx, commId2Txt, commId2articleID)                    
                    return AtopicWords4Epochs,CtopicWords4Epochs
                if f==0:
                    return AtopicWords4Epochs,CtopicWords4Epochs                

            
            if args.task == 'c':
                a,p,r,f,predictedLabel = classificationMain(args, modelOIVAE)
                print('a:{}, p: {}, r:{}, f:{}'.format(a,p,r,f))
                prfFile = os.path.join(e_dir, 'resultLog.txt')
                writePRF(prfFile,p,r,f,epoch,numTopic)
                if a > 0.4:
                    fn = os.path.join(t_dir, 'soccResult.txt')
                    writeSOCCAnswers(predictedLabel, testLabels, fn, Idx, commId2Txt, commId2articleID)
                    atwtensor = modelOIVAE.decoder.fcG1.weight.data
                    aTopicWords = getAtopic(atwtensor,10)
                    aTopicWords = getTopicWords(aTopicWords, iwD)
                    AtopicWords4Epochs.append(aTopicWords) 
                    ctwtensor = modelOIVAE.decoder.fcG2.weight.data
                    cTopicWords = getAtopic(ctwtensor,10)
                    cTopicWords = getTopicWords(cTopicWords, iwD)
                    CtopicWords4Epochs.append(cTopicWords)
                if f > 0.7:
                    return AtopicWords4Epochs,CtopicWords4Epochs
                


    parawriter = os.path.join(e_dir,'checkParameters.txt')
    info = 'final parameters of model when topic number is {}:\n'.format(numTopic)
    modelOIVAE.writeParas(writer=parawriter, statement=info)
    pbar.finish()
    
    return AtopicWords4Epochs,CtopicWords4Epochs

def runtrain(args, X, Y, mixedFile, iwD):
    modelList = []
    for numTopic in np.arange(args.numMinTopic, args.numMaxTopic, args.tStep):
        global t_dir
        t_dir = os.path.join(e_dir, 'topic_{}'.format(numTopic))
        if not os.path.exists(t_dir):
            os.makedirs(t_dir)
            
        modelVAE = standardVAE(net_arch=args, dimX=dimV, numTopic=numTopic)
        modelOIVAE = OIVAE(net_arch=args, dimX=dimV, numTopic=numTopic)
        discriminator = DISCRIMINATOR(net_arch=args, dimX=numTopic, dimH=20, dimC=2)
        if args.gpu:
            modelVAE = modelVAE.cuda() ; modelOIVAE = modelOIVAE.cuda() ; discriminator = discriminator.cuda()
        makeOptimizer(modelVAE, modelOIVAE, discriminator)
        
        parawriter = os.path.join(e_dir,'checkParameters.txt')
        info = 'initial parameters of model when topic number is {}:\n'.format(numTopic)
        modelOIVAE.writeParas(writer=parawriter, statement=info)
        
        AtopicWords4Epochs, CtopicWords4Epochs = train(args, modelVAE, modelOIVAE, discriminator, X, Y, iwD, numTopic)
        pk(os.path.join(t_dir,'AtopicWords4Epochs.pkl'),AtopicWords4Epochs)
        pk(os.path.join(t_dir,'CtopicWords4Epochs.pkl'),CtopicWords4Epochs)
        s1 = compareTopicChanging(AtopicWords4Epochs[0],AtopicWords4Epochs[-1])
        s2 = compareTopicChanging(CtopicWords4Epochs[0],CtopicWords4Epochs[-1])
        print('{}\n{}'.format(s1,s2))
#        s3 = compareTopicChanging(AtopicWords4Epochs[-1],CtopicWords4Epochs[-1])
#        print(s3)
        modelList.append(modelOIVAE)
        ###
#        for i, topicWords in enumerate(AtopicWords4Epochs):
#            writeTopics(topicWords, t_dir, i, info='aTopicWords')
#        for i, topicWords in enumerate(CtopicWords4Epochs):
#            writeTopics(topicWords, t_dir, i, info='cTopicWords')
    return modelList
###
    
def pair2clf(model, pair):
    tempX = pair2torch(pair)    
    tempP = model.getP(tempX, compute_loss=False)
    za = model.decoder.getZa(tempP)
    zc = model.decoder.getZc(tempP)
    #print(za.size(), zc.size())
    clf_xtrain = torch.cat((za,zc),dim=-1)
    return clf_xtrain

def trainClassifier(args, x_clf, y_clf, nTopic):
    classifier = DISCRIMINATOR(net_arch=args, dimX=2*(nTopic), dimH=nTopic, dimC=2)
    if args.gpu: classifier = classifier.cuda()
    if args.optimizer == 'Adam':
        optimizerCLF = torch.optim.Adam(classifier.parameters(), 0.001, betas=(args.mmt, 0.999))
    elif args.optimizer == 'SGD':
        optimizerCLF = torch.optim.SGD(classifier.parameters(), 0.001, momentum=args.mmt)
#    widgets = ['Classifier trainning: ', Percentage(), ' ', Bar(), ' ', ETA()]
#    pbar = ProgressBar(widgets=widgets, maxval=(args.numEpoch))
#    pbar.start()
    #classifier.train(True)
    for epoch in range(500):
        eIndices = torch.randperm(len(x_clf)).split(args.batchSize)
        for bi, bIndices in enumerate(eIndices):
            bx = x_clf[bIndices]
            by = y_clf[bIndices]
            optimizerCLF.zero_grad()
            clf_res, clf_loss = classifier(bx, by, compute_loss=True)
            clf_loss.backward(retain_graph=True)
            optimizerCLF.step()
#        pbar.update(epoch+1)
#    pbar.finish()
    return classifier

def classificationMain(args, model):
    
       

    m_dir = os.path.join(e_dir, 'topic_{}'.format(model.dimZ))
    accuracyTotal=0.0; precisionTotal=0.0; recallTotal=0.0; fTotal=0.0
    
    predictedLabel = torch.LongTensor([])
    for trainTime in range(len(Xtrains)):
        Xtrain = Xtrains[trainTime]
        Ytrain = Ytrains[trainTime]
        Xtest = Xtests[trainTime]
        Ytest = Ytests[trainTime] 
        testIdx = foldersIdx[trainTime]
        
        clf_xtrain = pair2clf(model, Xtrain)
        clf_ytrain = label2torch(Ytrain)
        clf_xtest = pair2clf(model, Xtest)     
        clf_ytest = label2torch(Ytest)                   
        classifier = trainClassifier(args, clf_xtrain, clf_ytrain, model.dimZ)
        a, p, r, f, predictedY = testClassification(trainTime, args, clf_xtest, clf_ytest, testIdx, classifier, m_dir)
        predictedLabel = torch.cat((predictedLabel,predictedY),dim=0)
        accuracyTotal += a ; precisionTotal += p ; recallTotal += r ; fTotal += f         
#            print('test {}: a:{}\tp:{}\tr:{}\tf:{}'.format(trainTime, a, p, r, f))          
    accuracyTotal = accuracyTotal/len(Xtrains)
    precisionTotal = precisionTotal/len(Xtrains)
    recallTotal = recallTotal/len(Xtrains)
    fTotal = fTotal/len(Xtrains)
    #print("the overall average a,p,r,f when topic number is {} are:\n {}, {}, {}, {}".format(model.dimZ, accuracyTotal, precisionTotal, recallTotal, fTotal))    
    return accuracyTotal, precisionTotal, recallTotal, fTotal, predictedLabel

def testClassification(trainTime, args, testX, testLabel, testIdx, classifier, m_dir):   
    #
    res = classifier(testX, testLabel, compute_loss=False)
    predictedY = predLabel(res)
    a, p, r, f = evaluateSOCC(predictedY, testLabel)
#    fn = os.path.join(m_dir, 'soccResult.txt')
#    writeSOCCAnswer(predictedY, testLabel, fn, trainTime, testIdx, commId2Txt, commId2articleID)
    return a, p, r, f, predictedY

#####
def testAlign(model):
    test_dir = os.path.join(e_dir, 'topic_{}'.format(model.dimZ))
    docIdComm = loadpk(os.path.join(data_dir, 'docIdComm.pkl'))
    articleId2Txt = loadpk(os.path.join(data_dir, 'articleId2Txt.pkl'))
    numComments = loadpk(os.path.join(data_dir, 'numComments.pkl'))
    
    M1 = model.decoder.fcM1.weight.data.clone() 
    M2 = model.decoder.fcM2.weight.data.clone()
    reversedM2 = torch.inverse(M2.clone())
    G1 = model.decoder.fcG1.weight.data.clone()
    G1 = torch.transpose(G1, -1, 0)
    G2 = model.decoder.fcG1.weight.data.clone()
    cwordProb = torch.transpose(G2, -1, 0) 
#    
    precisions=[]; recalls=[]; f1s=[]
    sm = nn.Softmax()
    
#    widgets = ['Testing: ', Percentage(), ' ', Bar(), ' ', ETA()]
#    pbar = ProgressBar(widgets=widgets, maxval=(sum(numComments.values())))
#    pbar.start()    
    testNum = 0

    for (articleID, commentIDs) in docIdComm.items():
        fff = True
        numComment = numComments[articleID]
        articleWithPeriod = articleId2Txt[articleID]
        answer = {}     
        for commentID in commentIDs:
            commentNum = commentID.strip().split('_')[1]
            commentIdx = commId2Idx[commentID]
            newdata = pair2torch([(articleID, commentID)])            
            zc = model.decoder.getZc(model.getP(newdata, compute_loss=False))
            zc = zeroOne(zc.view(model.dimZ))
            zc = zc.view(1, model.dimZ)
            za = zc.mm(reversedM2).mm(M1)
            za = sm(za)
            wordProb = za.mm(G1)  
            sentScore = {}
            sentences = articleWithPeriod.strip().split('<plp>')
            for j,sentence in enumerate(sentences):
                score = 0.0 #; wn = 0
                for word in sentence.strip().split(' '):
                    if word in wiDict:
                        wordScore = wordProb[0][wiDict[word]]
                        if wordScore>0:
                            score += wordScore #; wn+=1
                #wn = max(1, wn)
                #sentScore[j] = float(score/wn)
                sentScore[j] = float(score)
            commentScore = 0.0
            for cIdx in commentIdx:
                cwordScore = torch.sum(cwordProb[:,cIdx])
                commentScore += cwordScore
            threshold = commentScore/len(commentIdx)
            #r = sample([k for k in range(len(sentences))], 1)[0]
            #threshold = sentScore[r]*1.3
            #threshold = commentScore
            if fff:
                print(zc)
                print(za)
                print(threshold)
                print(sentScore)
                fff = False
            for j in range(len(sentScore)):
                if sentScore[j] < threshold:
                    pass
                else:
                    if commentNum in answer:
                        answer[commentNum].add(str(j+1))
                    else:
                        answer[commentNum] = set()
                        answer[commentNum].add(str(j+1))
#            pbar.update(testNum)
            testNum += 1
#        if fff:
#            print('-------\n', threshold, '\n', sentScore, '-------')
#            fff = False          
        precision, recall, f1 = evaluate(goldenAnswer=goldenAnswers[int(articleID)-1], answer=answer, numComment=numComment)
#            print('test on article {}:\nprecision: {}, recall: {}, f1: {}'.format(articleID, precision, recall, f1))
        #file = os.path.join(test_dir, 'YahooAnswer_t{}.txt'.format(model.dimZ))
        #writeYahooAnswer(articleID, goldenAnswers[int(articleID)-1], answer, numComment, file, commId2Txt)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
#        with open(os.path.join(output_dir, 'Performance.txt'), 'a') as pw:
#            pw.write('Number of Topic: {}\tArticle ID: {}\tp: {} r: {} f:{}\n '.format(model.dimZ, articleID, precision, recall, f1))
#        pw.close()
#    pbar.finish()
    avgP = sum(precisions)/len(precisions)
    avgR = sum(recalls)/len(recalls)
    avgF = sum(f1s)/len(f1s)
    
    return   avgP, avgR, avgF

def topicMain(args):
    if args.data == 'yahoo':
        modelList = runtrain(args, trainPairs, trainLabels, txtMerge, iwDict)
    elif args.data == 'socc':
        modelList = runtrain(args, testPairs, testLabels, txtMerge, iwDict)
    return modelList

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data', type=str, default='yahoo')
    parser.add_argument('-task', '--task', type=str, default='topic')
    parser.add_argument('-bs', '--batchSize', type=int, default=743)
    parser.add_argument('-hu1', '--hiddenUnit1', type=int, default=200)
    parser.add_argument('-hu2', '--hiddenUnit2', type=int, default=200)
    parser.add_argument('-mint', '--numMinTopic', type=int, default=5)
    parser.add_argument('-maxt', '--numMaxTopic', type=int, default=6)
    parser.add_argument('-ts', '--tStep', type=int, default=1)
    parser.add_argument('-ne', '--numEpoch', type=int, default=1000)
    parser.add_argument('-inim', '--initMult', type=int, default=1.0)
    parser.add_argument('-lam', '--lam', type=float, default= 1)
    parser.add_argument('-lr', '--lr', type=float, default=0.00001)
    parser.add_argument('-mmt', '--mmt', type=float, default=0.99)
    parser.add_argument('-v', '--variance', type=float, default=0.995)
    parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
    parser.add_argument('-gpu', '--gpu')        # do not use GPU acceleration
    args = parser.parse_args()

    cur_path = os.path.abspath(__file__)
    cur_dir = cur_path.split('/')[:-1]
    cur_dir = '/'.join(cur_dir)
    if args.data == 'yahoo':
        data_dir = os.path.join(cur_dir, 'data/yahoo')
        output_dir = os.path.join(cur_dir, 'output/yahoo')  
        p8 = os.path.join(data_dir, 'goldenAnswers.pkl')
        goldenAnswers = loadpk(p8)        
    elif args.data == 'socc':
        data_dir = os.path.join(cur_dir, 'data/socc')
        output_dir = os.path.join(cur_dir, 'output/socc')
        p13 = os.path.join(data_dir, 'commId2articleID.pkl')
        commId2articleID = loadpk(p13)
    else:
        raise ValueError

    global txtMerge,articleId2Idx,commId2Idx,trainPairs,testPairs,testLabels,wiDict,iwDict,dimV,commId2Txt
    txtMerge,articleId2Idx,commId2Idx,trainPairs,testPairs,testLabels,wiDict,iwDict,dimV,commId2Txt = loadGlobals(data_dir)
    trainLabels = torch.LongTensor(len(trainPairs)).fill_(0)
    e_dir = os.path.join(output_dir, 'e{}_lr{}_lam{}_mmt{}'.format(args.numEpoch,args.lr,args.lam,args.mmt)) 

    # default to use GPU, but have to check if GPU exists
    if args.gpu:
        if torch.cuda.device_count() == 0:
            args.gpu = False
            print('no GPU available.')
        else:
            print('use GPU')
    else:
        print('do NOT use GPU.')
        
    if args.task == 'topic':
        modelList = topicMain(args)
    elif args.task == 'a':
        modelList = topicMain(args)
#        for mth, model in enumerate(modelList):
#            print('test model {} of {}'.format(mth+1, len(modelList)))
#            testAlign(model)
    elif args.task == 'c':  
        global foldersIdx, Xtrains, Ytrains, Xtests, Ytests, Idx
        foldersIdx = []; Xtrains=[]; Ytrains=[]; Xtests=[]; Ytests=[]
        Idx = torch.randperm(len(testPairs))
        folders = Idx.split(int(len(testPairs)/10))
        for i,folder in enumerate(folders):
            otherFolders = crossValidation(folders, i)
            foldersIdx.append(folder)
            trainPairs_SOCC = batchPair(testPairs, otherFolders)
            trainLabels_SOCC = batchPair(testLabels, otherFolders)
            testPairs_SOCC = batchPair(testPairs, folder)
            testLabels_SOCC = batchPair(testLabels, folder)
            
            Xtests.append(testPairs_SOCC)
            Ytests.append(testLabels_SOCC)
            Xtrains.append(trainPairs_SOCC)
            Ytrains.append(trainLabels_SOCC)
        modelList = topicMain(args)




            # coherence
#            pathAtopicWords = os.path.join(cur_dir, 'tempAtopicWords.pkl')
#            pathCtopicWords = os.path.join(cur_dir, 'tempCtopicWords.pkl')
#            pk(pathAtopicWords, aTopicWords)
#            pk(pathCtopicWords, cTopicWords)
#            coherence = subprocess.getoutput("python ../coherence/coherence.py -path0 {} -path1 {} -path2 {}".format(cur_dir, pathAtopicWords, pathCtopicWords))
#            print(coherence, '\n\n')
#            acoherence, ccoherence = coherence.split('\t')
#            print(acoherence)
#            print(ccoherence)
#            a = float(acoherence.strip())
#            c = float(ccoherence.strip())
#            print(type(a), type(c))
#            print(a,c)
#            exit()



'''
def testAlign(model):
    test_dir = os.path.join(e_dir, 'topic_{}'.format(model.dimZ))
    docIdComm = loadpk(os.path.join(data_dir, 'docIdComm.pkl'))
    articleId2Txt = loadpk(os.path.join(data_dir, 'articleId2Txt.pkl'))
    numComments = loadpk(os.path.join(data_dir, 'numComments.pkl'))
    
    M1 = model.decoder.fcM1.weight.data.clone() 
    M2 = model.decoder.fcM2.weight.data.clone()
    reversedM2 = torch.inverse(M2.clone())
    G1 = model.decoder.fcG1.weight.data.clone()
    G1 = torch.transpose(G1, -1, 0)
    G2 = model.decoder.fcG1.weight.data.clone()
    cwordProb = torch.transpose(G2, -1, 0) 
#    
    precisions=[]; recalls=[]; f1s=[]
    
#    widgets = ['Testing: ', Percentage(), ' ', Bar(), ' ', ETA()]
#    pbar = ProgressBar(widgets=widgets, maxval=(sum(numComments.values())))
#    pbar.start()    
    testNum = 0
    fff = True
    for (articleID, commentIDs) in docIdComm.items():
        numComment = numComments[articleID]
        articleWithPeriod = articleId2Txt[articleID]
        answer = {}     
        for commentID in commentIDs:
            commentNum = commentID.strip().split('_')[1]
            commentIdx = commId2Idx[commentID]
            newdata = pair2torch([(articleID, commentID)])            
            zc = model.decoder.getZc(model.getP(newdata, compute_loss=False))
            zc = zeroOne(zc.view(model.dimZ))
            zc = zc.view(1, model.dimZ)
            za = zc.mm(reversedM2).mm(M1)
            wordProb = za.mm(G1)  
            sentScore = {}
            sentences = articleWithPeriod.strip().split('<plp>')
            for j,sentence in enumerate(sentences):
                score = 0.0 ; wn = 0
                for word in sentence.strip().split(' '):
                    if word in wiDict:
                        wordScore = wordProb[0][wiDict[word]]
                        score += wordScore ; wn+=1
                wn = max(1, wn)
                sentScore[j] = float(score/wn)
            commentScore = 0.0
            for cIdx in commentIdx:
                cwordScore = torch.sum(cwordProb[:,cIdx])
                commentScore += cwordScore
            threshold = commentScore/len(commentIdx)
            for j in range(len(sentScore)):
                if sentScore[j] < threshold:
                    pass
                else:
                    if commentNum in answer:
                        answer[commentNum].add(str(j+1))
                    else:
                        answer[commentNum] = set()
                        answer[commentNum].add(str(j+1))
#            pbar.update(testNum)
            testNum += 1
#        if fff:
#            print('-------\n', threshold, '\n', sentScore, '-------')
#            fff = False          
        precision, recall, f1 = evaluate(goldenAnswer=goldenAnswers[int(articleID)-1], answer=answer, numComment=numComment)
#            print('test on article {}:\nprecision: {}, recall: {}, f1: {}'.format(articleID, precision, recall, f1))
        file = os.path.join(test_dir, 'YahooAnswer_t{}.txt'.format(model.dimZ))
        writeYahooAnswer(articleID, goldenAnswers[int(articleID)-1], answer, numComment, file, commId2Txt)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        with open(os.path.join(output_dir, 'Performance.txt'), 'a') as pw:
            pw.write('Number of Topic: {}\tArticle ID: {}\tp: {} r: {} f:{}\n '.format(model.dimZ, articleID, precision, recall, f1))
        pw.close()
#    pbar.finish()
    if len(precisions) > 1:
        print("average p, r, f1 : {}, {}, {}".format(sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s) ))   
        with open(os.path.join(output_dir, 'Performance.txt'), 'a') as pw:
            pw.write('\nNumber of Topic: {} OVERALL PERFORMANCE:  p-{} r-{} f-{} \n\n\n'.format(
                    model.dimZ, sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1s)/len(f1s)))
        pw.close() 
    else:
        print('totally failed prediction!')
    
    return  
'''

'''for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
    #print('----')
def classificationMain(args, modelList):
    Xtrains=[]; Ytrains=[]; Xtests=[]; Ytests=[]
    global foldersIdx
    foldersIdx = []
    folders = torch.randperm(len(testPairs)).split(int(len(testPairs)/10))
    for i,folder in enumerate(folders):
        otherFolders = crossValidation(folders, i)
        foldersIdx.append(folder)
        trainPairs_SOCC = batchPair(testPairs, otherFolders)
        trainLabels_SOCC = batchPair(testLabels, otherFolders)
        testPairs_SOCC = batchPair(testPairs, folder)
        testLabels_SOCC = batchPair(testLabels, folder)
        
        Xtests.append(testPairs_SOCC)
        Ytests.append(testLabels_SOCC)
        Xtrains.append(trainPairs_SOCC)
        Ytrains.append(trainLabels_SOCC)
    #modelList = runtrain(args, Xtrain, Ytrain, txtMerge, iwDict)  #         
    
    for model in modelList:
        m_dir = os.path.join(e_dir, 'topic_{}'.format(model.dimZ))
        accuracyTotal=0.0; precisionTotal=0.0; recallTotal=0.0; fTotal=0.0
        for trainTime in range(len(Xtrains)):
            Xtrain = Xtrains[trainTime]
            Ytrain = Ytrains[trainTime]
            Xtest = Xtests[trainTime]
            Ytest = Ytests[trainTime] 
            testIdx = foldersIdx[trainTime]
            
            clf_xtrain = pair2clf(model, Xtrain)
            clf_ytrain = label2torch(Ytrain)
            clf_xtest = pair2clf(model, Xtest)     
            clf_ytest = label2torch(Ytest)                   
            classifier = trainClassifier(args, clf_xtrain, clf_ytrain, model.dimZ)
            a, p, r, f = testClassification(trainTime, args, clf_xtest, clf_ytest, testIdx, classifier, m_dir)
            accuracyTotal += a ; precisionTotal += p ; recallTotal += r ; fTotal += f         
#            print('test {}: a:{}\tp:{}\tr:{}\tf:{}'.format(trainTime, a, p, r, f))          
        accuracyTotal = accuracyTotal/len(Xtrains)
        precisionTotal = precisionTotal/len(Xtrains)
        recallTotal = recallTotal/len(Xtrains)
        fTotal = fTotal/len(Xtrains)
        print("the overall average a,p,r,f when topic number is {} are:\n {}, {}, {}, {}".format(model.dimZ, accuracyTotal, precisionTotal, recallTotal, fTotal))  
    
def testClassification(trainTime, args, testX, testLabel, testIdx, model):
    
    data = pair2torch(testX)
    target = label2torch(testLabel)
    z = model.getP(data, compute_loss=False)
    za = model.decoder.getZa(z)
    zc = model.decoder.getZc(z)
    #
    writeTensor(za, 'checkZAZC.txt')
    writeTensor(zc, 'checkZAZC.txt')
    
    #
    x_clf = torch.cat((za,zc),dim=-1)
    res = model.fcCLF(x_clf, target, compute_loss=False)
    predictedY = predLabel(res)
    a, p, r, f = evaluateSOCC(predictedY, testLabel)
    fn = os.path.join(t_dir, 'soccResult.txt')
    writeSOCCAnswer(predicted=predictedY, label=testLabel, file=fn, trainTime=trainTime, testIdx=testIdx, commId2Txt=commId2Txt)
    return a, p, r, f
'''
