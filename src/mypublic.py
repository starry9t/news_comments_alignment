#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:23:36 2019

@author: Yu Zhou
"""
import os
import re
import pickle
import torch
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger'
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['\'','per','pro','un','le','dr','-',"\"","(",")","–"])
stopwords = set(stop_words)
import enchant
endict = enchant.Dict("en_UK")
import gensim.corpora as corpora
from torch.autograd import Variable
from progressbar import Bar, ETA, FileTransferSpeed, Percentage, ProgressBar
from tensorboardX import SummaryWriter
from palmettopy.palmetto import Palmetto
from gensim.models.coherencemodel import CoherenceModel
from training import writeTWC

def loadGlobals(data_dir):
    txtMerge = os.path.join(data_dir, 'corpus.txt')
    p1 = os.path.join(data_dir, 'articleId2Idx.pkl')
    p2 = os.path.join(data_dir, 'commId2Idx.pkl')
    p3 = os.path.join(data_dir, 'trainPairs.pkl')
    p4 = os.path.join(data_dir, 'testPairs.pkl')
    p5 = os.path.join(data_dir, 'testLabels.pkl')
    p6 = os.path.join(data_dir, 'wiDict.pkl')
    p7 = os.path.join(data_dir, 'iwDict.pkl')
    p8 = os.path.join(data_dir, 'commId2Txt.pkl')
    
    articleId2Idx = loadpk(p1)
    commId2Idx = loadpk(p2)
    trainPairs = loadpk(p3)
    testPairs = loadpk(p4)
    testLabels = loadpk(p5)
    wiDict = loadpk(p6)
    iwDict = loadpk(p7)
    dimV = len(wiDict)
    commId2Txt = loadpk(p8)
    return   txtMerge, articleId2Idx,  commId2Idx, trainPairs, testPairs, testLabels, wiDict, iwDict, dimV, commId2Txt


def pk(file, data):
    with open(file, 'wb') as w:
        pickle.dump(data, w)
        w.close()
    return

def loadpk(file):
    with open(file, 'rb') as r:
        data = pickle.load(r)
        r.close()
    return data

delimiters = ' ','\t','\n',',','.','?','!','-','\'','’'
def mysplit(delimiters, string, maxsplit=0):
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

#mdelimiters = ' ','.'
#def mixsplit(mdelimiters=mdelimiters, string='', maxsplit=0):
#    regexPattern = '|'.join(map(re.escape, delimiters))
#    return re.split(regexPattern, string, maxsplit)    

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemSent(tokens):
    #tokens = word_tokenize(string)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) 
    s = " ".join(lemmas_sent)
    return s

def cleanStr(s):
    strs = mysplit(delimiters=delimiters, string=s, maxsplit=0)
    lowstrs = []
    for astr in strs:
        if astr:
            lowstrs.append(astr.lower())
        else:
            pass
    s = lemSent(lowstrs)
    newstring = []
    for astr in s.split(' '):
        try:
            if len(astr) == 1:
                continue
            elif astr not in stopwords:
                if endict.check(astr):
                    newstring.append(astr)
        except:
            pass
    ns = ' '.join(newstring)
    return ns, len(ns)

#s = 'The former House speaker, who oversaw passage of the Defense of Marriage Act in Congress and helped finance state campaigns to fight gay marriage in 2010, said in a Huffington Post interview that the party should work toward acceptance of rights for gay couples, while still distinguishing them from marriage.'
#
    
#s = 'Treinforce party’s wariness gun limit'
#ns, la = cleanStr(s)
#print(ns)


def compareTopicChanging(topicA, topicB):
    total = len(topicA)*len(topicA[0])
    wordsA = set() ; wordsB = set()
    for topicWords in topicA:
        for word in topicWords:
            wordsA.add(word)
    for topicWords in topicB:
        for word in topicWords:
            wordsB.add(word)
    mutual = len(wordsA&wordsB)
    s = 'there are {} same words of {} words in these two topics.'.format(mutual, total)  
    return s

def removePeriod(article):
    new = ''
    sents = article.strip().split('<plp>')
    for sent in sents:
        new += '{} '.format(sent.strip())
    return new.strip()

def buildDict(txtFile, d={}, tt='a'):
    with open(txtFile, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            try:
                if tt == 'a':
                    line = line.strip().split('\t')[1]
                elif tt == 'c':
                    line = line.strip().split('\t')[3]
                elif tt == 'ya':
                    line = line.strip().split('\t')[1]
                    line = removePeriod(line)
                elif tt == 'yc':
                    line = line.strip().split('\t')[2]
                else:
                    print('txtFile type error.(should be a or c)')
                    raise TypeError
            except:
                print(line)
                line = ''
            words = line.strip().split(' ')
            for word in words:
                if word in d:
                    d[word] += 1
                else:
                    d[word] = 1
    reader.close()
    return d
    
def removeFreq(d, high=99999, low=0):
    finalDict = {}
    for (word, freq) in d.items():
        if freq > high:
            continue
        if freq < low:
            continue
        finalDict[word] = freq   
    if '.' in finalDict:
        del finalDict['.']
    return finalDict

def removePercent(d, p1=0.25, p2=0.4):
    total = len(d)
    k1 = int(total*p1) ; k2 = int(total*p2)
    f = sorted(d.items(), key=lambda item:item[1], reverse=True)
    for (word, freq) in f[:k1]:
        del d[word]
    for (word, freq) in f[-k2:]:
        del d[word]        
    return d

def showDictWords(d, k=10):                   
    f = sorted(d.items(), key=lambda item:item[1], reverse=True)
    print("top {} words:\n{}\nbottom 20 words:\n{}".format(k, f[:k],f[-k:]))                

def biDict(d):
    n = 0; wiDict={}; iwDict={}
    for word in d.keys():
        wiDict[word] = n
        iwDict[n] = word
        n += 1
    wiDict['nullnull'] = n
    iwDict[n] = 'nullnull' 
    return wiDict,iwDict    

def mergeDict(d1, d2):
    for (word, freq) in d1.items():
        if word in d2:
            d2[word] += freq
        else:
            d2[word] = freq
    return d2

def writeCoherence(topicWords4Epochs, path, mixedFile='', command=[False,False], info=''):
    file = os.path.join(path, '{}.txt'.format(info))
    if command[0]:
        CohWriterCV = SummaryWriter(os.path.join(path,'runs/coh{}_cv'.format(info)))
        corpus, text, id2word = buildCorpusDict(mixedFile)  
    if command[1]:
        CohWriterUmass = SummaryWriter(os.path.join(path,'runs/coh{}_umass'.format(info)))
        pmt = Palmetto() 
    widgets = ['writing {}: '.format(info), Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=(len(topicWords4Epochs)))
    pbar.start()    
        
    for i, topicWords in enumerate(topicWords4Epochs):
        if command[0]:
            try:
                cm = CoherenceModel(topics=topicWords, corpus=corpus, texts=text, dictionary=id2word, coherence='c_v')
                coherence = cm.get_coherence()
                CohWriterCV.add_scalar('coherenceCV', coherence, i)
                coherences = cm.get_coherence_per_topic() 
            except:
                coherence=0; coherences=0
        else:
            coherence=0; coherences=[0 for i in range(len(topicWords))]
        if command[1]:
            try:
                coherences2 = []
                for topic in topicWords:
                    coherence = pmt(topic, coherence_type='umass')
                    coherences2.append(coherence)
                coherence2 = sum(coherences2)/len(coherences2)
                CohWriterUmass.add_scalar('coherenceUMASS', coherence2 , i)
                coherenceList = [coherences, coherence, coherences2, coherence2]
            except:
                coherence2=0; coherences2=[0 for i in range(len(topicWords))]
        else:
            coherence2=0; coherences2=0
        coherenceList = [coherences, coherence, coherences2, coherence2]
        writeTWC(topicWords, coherenceList, file, 'article', i, command)
        pbar.update(i+1)
    pbar.finish()
    return

def writeTopics(topicWords, path, epoch, coherences, info=''):
    file = os.path.join(path, '{}.txt'.format(info))
    with open(file, 'a') as w:
        w.write('topic in Epoch {}:\n'.format(epoch))
        for i, words in enumerate(topicWords):
            w.write('{}\t'.format(coherences[i]))
            for word in words:
                w.write('{}\t'.format(word))
            w.write('\n')
        w.write('\n\n\n')
    w.close()
    return
    
    


def makeSoccData(aFile, cFile, trainFile, pdir):
    #global iwDict, wiDict, X, Y, articleIdTxt, commentIdTxt, commentIdLabel, articleIdComm
    # articleIdTxt
    with open(aFile, 'r') as ar:
        lines = ar.readlines()
    ar.close()
    articleIdTxt = {}
    for line in lines:
        articleID, article = line.strip().split('\t')
        articleIdTxt[articleID] = article
    pk(os.path.join(pdir,'articleIdTxt.pkl'),articleIdTxt)
    # 
    with open(cFile, 'r') as cr:
        lines = cr.readlines()
    cr.close()
    commentIdTxt = {}
#    commentIdLabel = {}
    pseudoX = []
    
    for line in lines:
        articleID, commentID, cLabel, comment = line.strip().split('\t')
        commentIdTxt[commentID] = comment
#        commentIdLabel[commentID] = convL(cLabel)
        pseudoX.append((articleID, commentID, convL(cLabel)))
    pk(os.path.join(pdir,'commentIdTxt.pkl'),commentIdTxt)
    pk(os.path.join(pdir,'pseudoX.pkl'),pseudoX)
    
    with open(trainFile, 'w') as w:
        for (articleID, commentID, label) in pseudoX:
            article = articleIdTxt[articleID]
            comment = commentIdTxt[commentID]
            w.write('{}\t{} {}\n'.format(label, article.strip(), comment.strip()))
    w.close()

    corpus, data, id2word = buildCorpusDict(trainFile)
    pk(os.path.join(pdir,'Cohcorpus.pkl'),corpus)
    pk(os.path.join(pdir,'Cohdata.pkl'),data)
    pk(os.path.join(pdir,'Cohid2word.pkl'),id2word)

def convL(s):
    if s=='yes':
        return 1
    elif s=='no':
        return 0
    else:
        print('label beyond yes or no')
        raise NameError('label error!')
        
def buildCorpusDict(dataFile):
    data = []
    with open(dataFile, 'r') as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip().split('\t')[1]
            words = line.strip().split(' ')
            data.append(words)
    r.close()
    id2word = corpora.Dictionary(data)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]
    return corpus, data, id2word

def batchPair(pairs, idx):
    bp = []
    for index in idx:
        bp.append(pairs[index])
    return bp
    
#def makeYahooData(aFile, cFile, trainFile, pdir):
#    #global iwDict, wiDict, X, Y, articleIdTxt, commentIdTxt, commentIdLabel, articleIdComm
#    # articleIdTxt
#    with open(aFile, 'r') as ar:
#        lines = ar.readlines()
#    ar.close()
#    articleIdTxt = {}
#    for line in lines:
#        articleID, article = line.strip().split('\t')
#        articleIdTxt[articleID] = article
#    pk(os.path.join(pdir,'articleIdTxt.pkl'),articleIdTxt)
#    # 
#    with open(cFile, 'r') as cr:
#        lines = cr.readlines()
#    cr.close()
#    commentIdTxt = {}
##    commentIdLabel = {}
#    pseudoX = []
#    
#    for line in lines:
#        articleID, commentID, comment = line.strip().split('\t')
#        commentIdTxt[commentID] = comment
##        commentIdLabel[commentID] = convL(cLabel)
#        pseudoX.append((articleID, commentID, sample([0,1],1)[0]))
#    pk(os.path.join(pdir,'commentIdTxt.pkl'),commentIdTxt)
#    pk(os.path.join(pdir,'pseudoX.pkl'),pseudoX)
#    
#    with open(trainFile, 'w') as w:
#        for (articleID, commentID, label) in pseudoX:
#            article = articleIdTxt[articleID]
#            comment = commentIdTxt[commentID]
#            w.write('{}\t{} {}\n'.format(label, article.strip(), comment.strip()))
#    w.close()
#    
#    finalDict, wiDict, iwDict = buildDicts(txtFiles=[aFile,cFile], high=600, low=5)
#    pk(os.path.join(pdir,'finalDict.pkl'),finalDict)
#    pk(os.path.join(pdir,'wiDict.pkl'),wiDict)
#    pk(os.path.join(pdir,'iwDict.pkl'),iwDict)
#
#    corpus, data, id2word = buildCorpusDict(trainFile)
#    pk(os.path.join(pdir,'Cohcorpus.pkl'),corpus)
#    pk(os.path.join(pdir,'Cohdata.pkl'),data)
#    pk(os.path.join(pdir,'Cohid2word.pkl'),id2word)

def removeTrain(folds,i):
    for j in range(len(folds)):
        if j != i:
            idx = folds[j]
            k = j
            break
        else:
            continue
    for m in range(len(folds)):
        if m!=i and m!=k:
            idx = torch.cat((idx, folds[m]), dim=-1)
    return idx


def convResult(resultP):
    result = []
    for row in resultP:
        if row[0] > row[1]:
            result.append(0)
        else:
            result.append(1)
    return result

def writeResult(result, testY, accuracy, precision, recall, f1, file):
    with open(file, 'a') as w:
        w.write('accuracy:{}, precision:{}, recall:{}, f1-value:{}'.format(accuracy, precision, recall, f1))
        w.write('predicted\ttrue label\n')
        for i in range(len(testY)):
            w.write('{}\t{}\n'.format(result[i],testY[i]))
    w.close()

def writeTensor(tensor, file):
    with open(file, 'a') as w:
        d = len(tensor.size())
        for i in range(d):
            row = tensor[i]
            for number in row:
                w.write('{}\t'.format(number))
            w.write('\n')
        w.write('\n\n')
        w.write('-----------------------------------------------------\n')
        w.close()
        
def predLabel(pred):
    l = []
    for i in range(len(pred)):
        if pred[i][0] > pred[i][1]:
            l.append(0)
        else:
            l.append(1)
    newpred = torch.tensor(l)
    return newpred

def crossValidation(folders, i):
    t = torch.LongTensor([])
    for m in range(len(folders)):
        if m != i:
            t = torch.cat((t,folders[m]), dim=-1)
        else:
            pass
    return t
        
###############################################################################
def oldbuildDataMatrix(file, wiDict):
    wlen = len(wiDict)
    with open(file, 'r') as r:
        lines = r.readlines()
    r.close()
    data = np.zeros((int(len(lines)-1), wlen))
    print("initialize data as {}".format(data.shape))
    
    for i,line in enumerate(lines[:-1]):
        text = line.strip().split(' ')
        for word in text:
            if word in wiDict:
                data[i, wiDict[word]] += 1
    return data

def oldbuildDicts(txtFile='', high=500, low=5):
    fullDict = {}
    finalDict = {}
    with open(txtFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split(' ')
            for word in words:
                if word in fullDict:
                    fullDict[word] += 1
                else:
                    fullDict[word] = 1
    f.close()
    for (word, freq) in fullDict.items():
        if freq > high:
            continue
        if freq < low:
            continue
        finalDict[word] = freq
    print("there are {} words in finalDict.".format(len(finalDict)))
    print("there are {} words in fullDict.".format(len(fullDict)))
    f = sorted(finalDict.items(), key=lambda item:item[1], reverse=True)
    print("top 20 words:\n{}\nbottom 20 words:\n{}".format(f[:20],f[-20:]))
    wiDict = {}
    iwDict = {}
    n = 0
    for word in finalDict.keys():
        wiDict[word] = n
        iwDict[n] = word
        n += 1
    return finalDict,wiDict,iwDict































