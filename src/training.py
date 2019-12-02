# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:54:26 2019

@author: zhouyu
"""
import os
import torch
import random
import numpy as np
import gensim.corpora as corpora
import torch.nn as nn
import time

def getAtopic(twtensor, k=10):
    topicWords = []
    twtensor = torch.transpose(twtensor, -1, 0)
    for line in twtensor:
        (prob, wordIdx) = torch.topk(line, k)
        tw = [int(i) for i in wordIdx]
        topicWords.append(tw)
    return topicWords

def getLoss(x, y, miu, sigma):
    residual = torch.abs(y - x)
    reLoss = torch.sqrt(torch.sum(torch.pow(residual, 2))) / (x.size(0))
#    lossF = nn.MSELoss(reduce=True,reduction='mean')
#    reLoss = lossF(x,y.type(torch.FloatTensor))
    amiu = miu.clone()
    asigma = sigma.clone()
    KLLoss = 0.5*torch.sum(1 + torch.log(torch.clamp(torch.pow(asigma,2), min=1e-10)) - torch.pow(amiu,2) - torch.pow(asigma,2)) / x.size(0)
    return reLoss, KLLoss, reLoss-KLLoss

def buildCorpusDict(dataFile):
    data = []
    with open(dataFile, 'r') as r:
        lines = r.readlines()
        for line in lines:
            words = line.strip().split(' ')
            data.append(words)
    r.close()
    id2word = corpora.Dictionary(data)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]
    return corpus, data, id2word

def getTopicWords(topicWords, iwDict):
    return [[iwDict[i] for i in topic] for topic in topicWords]

def loadLabel(k):
    return [random.choice([0,1]) for i in range(k)]

#def drawTBoard(ys, Xaxis, tag):
#    writer = SummaryWriter('runs/{}'.format(tag))
#    for y, x in zip(ys, range(Xaxis)):
#        writer.add_scalar(tag, y, x)
#    return
    
def getTestDatas(articleFile, commentFile):
    
    articleIdTxt = {}
    with open(articleFile, 'r') as r:
        lines = r.readlines()
        for line in lines:
            articleId, article = line.strip().split('\t')
            articleIdTxt[articleId] = article
    r.close()
    
    docIdComm = {}; commentIdTxt = {}; numComments = {}
    with open(commentFile, 'r') as r:
        lines = r.readlines()
        for line in lines:
            articleId, commentId, comment = line.strip().split('\t')
            commentIdTxt[commentId] = comment
            if articleId in docIdComm:
                docIdComm[articleId].append(commentId)
                numComments[articleId] += 1
            else:
                docIdComm[articleId] = [commentId]
                numComments[articleId] = 1
    r.close()
    return articleIdTxt, docIdComm, commentIdTxt, numComments

def zeroOne(tensor):
    ctensor = tensor.clone()
    total = torch.sum(ctensor)
#    print(total)
    if total == 0:
        total = 1
    for i in range(tensor.size(0)):
        tensor[i] = ctensor[i]/total
#    print(tensor)
    (prob, wordIdx) = torch.topk(tensor, tensor.size(0))
#    print(prob, '\n', wordIdx)
    oneIdx = [wordIdx[0]]
#    print(oneIdx)
    for j in range(tensor.size(0)-1):
        t = ctensor[wordIdx[j]]/ctensor[wordIdx[j+1]]
        if t>1.5 or t<0:
            break
        else:
            oneIdx.append(wordIdx[j+1])
            pass
#    print(oneIdx)
    for k in range(tensor.size(0)):
        if k in oneIdx:
            tensor[k] = 1
        else:
            tensor[k] = 0
#    print(tensor)
    return tensor

#c = torch.tensor([-0.2332,  0.0870,  0.4688, -0.0424, -0.4835, -0.1794, -0.1405,  0.4126, 0.3226,  0.2810,  0.1691, -0.1661,  0.2420,  0.0618, -0.3139, -0.3127, 0.4242, -0.2955,  0.2648, -0.1778])
#print(c.size())
#print(c.view(20).size(0))
#cc = zeroOne(c.view(20))
#print(cc)

def evaluate(goldenAnswer, answer, numComment):
    if len(answer) == 0:
        #print("prediction failed!!!\n")
        return 0,0,0
    numRight = 0
    for i in range(numComment):
        k = str(i+1)
        if k in goldenAnswer and k in answer:
            right = goldenAnswer[k]
            predict = answer[k]
            if right&predict:
                numRight += 1
            else:
                pass
#        elif k not in goldenAnswer and k not in answer:
#            numRight += 1
        
    if numRight==0:
        return 0,0,0    
    precision = numRight/len(answer)
    recall = numRight/len(goldenAnswer)
    f1 = (2*precision*recall)/(precision+recall)
    return precision, recall, f1

def evaluateSOCC(predicted, label):
    assert len(predicted) == len(label)
    total = len(label)
    tp=0; fp=0; fn=0; tn=0
    for i in range(total):
        if predicted[i]==1 and label[i]==1:
            tp += 1
        elif predicted[i]==1 and label[i]==0:
            fp += 1
        elif predicted[i]==0 and label[i]==1:
            fn += 1
        elif predicted[i]==0 and label[i]==0:
            tn += 1
        else:
            raise ValueError
    if tp == 0:
        #print('ding! all wrong!')
        return 0,0,0,0
    accuracy = (tp+tn)/total
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return accuracy, precision, recall, f1

def writePRF(file, p=0.0,r=0.0,f=0.0,epoch=1,numTopic=5):
    with open(file, 'a') as w:
        w.write('{}:\t'.format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
        w.write('p:{},r:{},f:{},epoch:{},numTopic:{}\n'.format(p,r,f,epoch,numTopic))
    w.close()
    return
def writeYahooAnswer(articleID, goldenAnswer, answer, numComment, file, commId2Txt):
    with open(file, 'a') as w:
        w.write('Answer of Article {}\tlabel\tpredicted\n'.format(articleID))
        for i in range(numComment):
            commID = '{}_{}'.format(articleID,i+1)
            #w.write('{}\t'.format(commID))
            k = str(i+1)
            if k in goldenAnswer:
                if k in answer:
                    w.write('{}:\t{}\t{}'.format(commID,goldenAnswer[k],answer[k]))
                else:
                    w.write('{}:\t{}\tempty'.format(commID,goldenAnswer[k]))
            else:
                if k in answer:
                    w.write('{}:\tempty\t{}'.format(commID,answer[k]))
                else:
                    w.write('{}:\temtpy\tempty'.format(commID))
            w.write('\t----{}\n'.format(commId2Txt[commID]))
        w.write('\n\n')
    w.close()
    return

def writeSOCCAnswer(predicted, label, file, trainTime, testIdx, commId2Txt, commId2articleID):
    with open(file, 'a') as w:
        w.write('test set {}:\n'.format(trainTime))
        w.write('aId\tpredicted\tlabel\tcomment\n')
        j = 0
        for pred, y in zip(predicted, label):
            idx = testIdx[j]
            w.write('{}\t{}\t{}\t{}\n'.format(commId2articleID[idx],pred,y,commId2Txt[int(idx)]))
            j += 1
        w.write('\n\n')
    w.close()
    
def writeSOCCAnswers(predicted, label, file, Idx, commId2Txt, commId2articleID):
    #print(len(predicted), len(label), len(Idx), len(commId2articleID), len(commId2Txt))
    with open(file, 'a') as w:
        w.write('aId\tpredicted\tlabel\tcomment\n')
        j = 0
        for pred, y in zip(predicted, label):
            idx = int(Idx[j])
            w.write('{}\t{}\t{}\t{}\n'.format(commId2articleID[idx],pred,y,commId2Txt[idx]))
            j += 1
        w.write('\n\n')
    w.close()

def getTTwords(aTopicWord, cTopicWord, model, iwDict, file, countN):    
    m1 = model.fcM1.weight.data
    m2 = model.fcM2.weight.data
    g1 = model.fcG1.weight.data
    g2 = model.fcG2.weight.data
    m = torch.mm(torch.inverse(m2),m1)
    m = m.detach().numpy()
    numTopic = len(aTopicWord)
#    print('numTopic: {}'.format(numTopic))
#    print('m.shape: {}'.format(m.shape))
#    print('length of iwDict: {}'.format(len(iwDict)))
#    print('shape of atw and ctw: {}*{}, {}*{}'.format(len(aTopicWord),len(aTopicWord[0]), len(cTopicWord), len(cTopicWord[0])))
    with open(file, 'a') as f:
        f.write('topic relation in Epoch {}:'.format(countN*50))
        for i in range(numTopic):
            atwords = [iwDict[ind] for ind in aTopicWord[i]]
            f.write("Topic {} : {}\n".format(i, atwords))
            cTopicList = list(m[:,i])
            cTopicInd = cTopicList.index(max(cTopicList))
            ctwords = cTopicWord[cTopicInd]
            ctwords = [iwDict[int(ind)] for ind in ctwords]
            f.write("Attract topic: {}\n\n".format(ctwords))
    f.close()        
    return

def writeTWC(topicWords, coherenceList, file, tag, epochNum, command=[True, True]):
    coherences, coherence, coherences2, coherence2 = coherenceList[:]
    with open(file, 'a') as w:
        if command[0]:
            w.write('{} topic in Epoch {}: (average coherence: {}\t\'c_v\')\n'.format(tag, ((epochNum+1)), coherence,))
            idxs = list(np.argsort(-(np.array(coherences))))
            for idx in idxs:
                w.write('{}\t'.format(coherences[idx]))
                for word in topicWords[idx]:
                    w.write('{}\t'.format(word))
                w.write('\n')
            w.write('\n')
        if command[1]:
            w.write('{} topic in Epoch {}: (average coherence: {}\t\'u_mass\')\n'.format(tag, ((epochNum+1)), coherence2,))
            idxs = list(np.argsort(-(np.array(coherences2))))
            for idx in idxs:
                w.write('{}\t'.format(coherences2[idx]))
                for word in topicWords[idx]:
                    w.write('{}\t'.format(word))
                w.write('\n')
        w.write('\n\n\n')
    w.close()
    return

def writeParameters(odir, model, fidx, args, numTopic, tag):
    with open(os.path.join(odir,'twTXT{}/e{}_lr{}_lam{}_mmt{}_checkParameters.txt'.format(fidx, args.numEpoch,args.lr,args.lam,args.mmt)), 'a') as w:
        w.write('{} parameters of model when topic number is {}:\n'.format(tag, numTopic))
        w.write('fcEn1:\n{}\n\nfcEn2:\n{}\n\nfcMiu:\n{}\n\nfcSigma:\n{}\n\nfcM1:\n{}\n\nfcM2:\n{}\n\nfcG1:\n{}\n\nfcG2:\n{}\n\n\n\n'.format(
                model.fcEn1.weight.data, model.fcEn2.weight.data, model.fcMiu.weight.data,model.fcSigma.weight.data, 
                model.fcM1.weight.data, model.fcM2.weight.data, model.fcG1.weight.data, model.fcG2.weight.data))
    w.close()
    return