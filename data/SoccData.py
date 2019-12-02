# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:49:40 2019

@author: zhouyu
"""

import os
import argparse
import sys
sys.path.append(r"/home/zhouyu/github/acvae/src")
import pickle
import csv
from progressbar import Bar, ETA, FileTransferSpeed, Percentage, ProgressBar
from mypublic import pk, loadpk, cleanStr, buildDict, removeFreq, removePercent, showDictWords, biDict, mergeDict

cur_dir = os.getcwd()
o_dir = os.path.join(cur_dir, "socc/original")
data_dir = os.path.join(cur_dir, "socc")

csvArticles = os.path.join(o_dir, 'gnm_articles.csv')
csvComments = os.path.join(o_dir, 'gnm_comments.csv')
csvConstructive = os.path.join(o_dir, 'constructive_comments.csv')

txtArticles = os.path.join(data_dir, 'articles.txt')
txtComments = os.path.join(data_dir, 'comments.txt')
txtMerge = os.path.join(data_dir, 'corpus.txt')
testA = os.path.join(data_dir, 'Tarticles.txt')
testC = os.path.join(data_dir, 'Tcomments.txt')

p1 = os.path.join(data_dir, 'articleId2Idx.pkl')
p2 = os.path.join(data_dir, 'commId2Idx.pkl')
p3 = os.path.join(data_dir, 'trainPairs.pkl')
p4 = os.path.join(data_dir, 'testPairs.pkl')
p5 = os.path.join(data_dir, 'testLabels.pkl')
p6 = os.path.join(data_dir, 'wiDict.pkl')
p7 = os.path.join(data_dir, 'iwDict.pkl')

p9 = os.path.join(data_dir, 'commId2Txt.pkl')
p13 = os.path.join(data_dir, 'commId2articleID.pkl')

def convertLabel(label):
    if label == 'yes':
        return 1
    elif label == 'no':
        return 0 
    else:
        return 'UNKNOWN'

def getTestFile():
    testArticleSet = set(); trainArticleSet = set()
    testCommentSet = set()#; trainCommentSet = set()
    commId2Txt = {} ; commId2articleID = {}
    widgets = ['Writing: ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=1000000)
    pbar.start()
    p = 0; pp=0; ppp=0
    #
    with open(csvConstructive, 'r', errors='ignore') as f:
        reader = csv.DictReader(f)
        with open(testC, 'w') as w:
            idx = 0
            for row in reader:
                p += 1
                articleID = row['article_id']
                commentID = row['comment_counter']
                comment = row['comment_text']
                commentLabel = row['is_constructive']
                commId2Txt[idx] = comment
                commId2articleID[idx] = articleID
                idx+=1
                comment, la = cleanStr(comment)      
                testArticleSet.add(articleID)
                testCommentSet.add(commentID)
                w.write('{}\t{}\t{}\t{}\n'.format(articleID, commentID, convertLabel(commentLabel), comment)) 
                pbar.update(p)
                #no matter if comment is empty string
        w.close()
    f.close()   
    print('there are {} articles involved in test set.  {}'.format(len(testArticleSet), p))
    #
    with open(csvComments, 'r', errors='ignore') as f:
        reader = csv.DictReader(f)
        with open(txtComments, 'w') as w:
            for row in reader:
                pp += 1
                articleID = row['article_id']
                commentID = row['comment_counter']
                comment = row['comment_text']
                comment, la = cleanStr(comment)
                pbar.update(p+pp)
                if commentID in testCommentSet:
                    continue
                else:
                    if la > 5:
                        w.write('{}\t{}\t{}\t{}\n'.format(articleID, commentID, commentLabel, comment))
                        trainArticleSet.add(articleID)
            w.close()
    f.close()   
    print('there are {} articles involved in train set. {}'.format(len(trainArticleSet), pp))
    
    total = 0; lost = 0;
    with open(csvArticles, 'r') as f:
        reader = csv.DictReader(f)
        with open(testA, 'w') as testw:
            with open(txtArticles, 'w') as trainw:
                for row in reader:
                    ppp += 1
                    total += 1
                    articleID = row['article_id']
                    article = row['article_text']
                    article, la = cleanStr(article)
                    pbar.update(p+pp+ppp)
                    if articleID in trainArticleSet:
                        trainw.write('{}\t{}\n'.format(articleID, article))
                    elif articleID in testArticleSet:
                        testw.write('{}\t{}\n'.format(articleID, article))
                    else:
                        #print('article {} has no comment.'.format(articleID))
                        lost += 1
            trainw.close()
        testw.close()
    f.close()
    print(ppp)
    pbar.finish()
    print('there are {} articles in total and {} articles removed because of no comment.'.format(total, lost))
    
    pk(p9,commId2Txt)
    pk(p13, commId2articleID)
    return

def csv2txt():
    getTestFile()
###############################################################################   
    
def getDicts():
    aDict1 = buildDict(d={}, txtFile=txtArticles, tt='a')
    aDict2 = buildDict(d={}, txtFile=testA, tt='a')
    aDict = mergeDict(aDict1, aDict2) ; l1 = len(aDict)
    #aDict = removeFreq(d=aDict, high=4000, low=50) ; l2 = len(aDict)
    aDict = removePercent(d=aDict, p1=0.15, p2=0.55) ; l2 = len(aDict)
    print('aDict from {} words to {} words.'.format(l1, l2))
#    showDictWords(aDict,20)
    cDict1 = buildDict(d={}, txtFile=txtComments, tt='c')
    cDict2 = buildDict(d={}, txtFile=testC, tt='c')
    cDict = mergeDict(cDict1, cDict2) ; l3 = len(cDict)
    #cDict = removeFreq(d=cDict, high=10000, low=500) ; l4 = len(cDict)
    cDict = removePercent(d=cDict, p1=0.15, p2=0.65) ; l4 = len(cDict)
    print('cDict from {} words to {} words.'.format(l3, l4))
#    showDictWords(cDict,20)
    finalDict = mergeDict(aDict, cDict) ; print('{}+{} ?= {}'.format(l2, l4, len(finalDict)))
    wiDict, iwDict = biDict(finalDict)
#    print('finalDict')
#    showDictWords(finalDict)
    return aDict, cDict, finalDict, wiDict, iwDict

def getMappings(aDict, cDict, finalDict, wiDict, iwDict):
    articleId2Idx = {} ; commId2Idx = {} ;   commId2Txt = {}
    trainPairs = [] ; testPairs = [] ; testLabels = []
    #
    with open(txtComments, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            articleID, commentID, label, comment = line.strip().split('\t')
            comm = comment.strip().split(' ') ; commIdx = []
            for word in comm:
                if word in cDict:
                    commIdx.append(wiDict[word])
            if len(commIdx) > 5:
                trainPairs.append((articleID, commentID))
                commId2Idx[commentID] = commIdx        
    reader.close()
    print('1')
    with open(txtArticles, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            articleID, article = line.strip().split('\t')
            atc = article.strip().split(' ') ; atcIdx = []
            for word in atc:
                if word in aDict:
                    atcIdx.append(wiDict[word])
            articleId2Idx[articleID] = atcIdx
    reader.close()
    print('2')
    with open(testC, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            try:
                [articleID, commentID, label, comment] = line.strip().split('\t')
            except:
                print(line)
            comm = comment.strip().split(' ') ; commIdx = []
            for word in comm:
                if word in cDict:
                    commIdx.append(wiDict[word])
            testPairs.append((articleID, commentID))
            testLabels.append(int(label))
            commId2Idx[commentID] = commIdx        
    reader.close()
    print('3')
    with open(testA, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            articleID, article = line.strip().split('\t')
            atc = article.strip().split(' ') ; atcIdx = []
            for word in atc:
                if word in aDict:
                    atcIdx.append(wiDict[word])
            articleId2Idx[articleID] = atcIdx
    reader.close()
    print('4')
    # save
    pk(p1,articleId2Idx)
    pk(p2,commId2Idx)
    pk(p3,trainPairs)
    pk(p4,testPairs)
    pk(p5,testLabels)
    pk(p6,wiDict)
    pk(p7,iwDict)
    
    
def reWriteTxt():
    aDict, cDict, finalDict, wiDict, iwDict = getDicts()
    getMappings(aDict, cDict, finalDict, wiDict, iwDict)
    
    
def run(args):
    if args.cmd == 'c2t':
        csv2txt()
        
    if args.cmd == 't2t':
        reWriteTxt()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cmd', '--cmd', type=str, default='c2t')
    args = parser.parse_args()
    run(args)