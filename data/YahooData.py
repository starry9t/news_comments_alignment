# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:50:36 2019

@author: zhouyu
"""
import os
import argparse
import sys
sys.path.append(r"/home/zhouyu/github/acvae/src")
import pickle
from collections import defaultdict
from mypublic import pk, loadpk, cleanStr, buildDict, removePeriod, removeFreq, removePercent, showDictWords, biDict, mergeDict

cur_dir = os.getcwd()
o_dir = os.path.join(cur_dir, "yahoo/original/en")
data_dir = os.path.join(cur_dir, "yahoo")

txtArticles = os.path.join(data_dir, 'articles.txt')
txtComments = os.path.join(data_dir, 'comments.txt')
txtMerge = os.path.join(data_dir, 'corpus.txt')

p1 = os.path.join(data_dir, 'articleId2Idx.pkl')
p2 = os.path.join(data_dir, 'commId2Idx.pkl')
p3 = os.path.join(data_dir, 'trainPairs.pkl')
p4 = os.path.join(data_dir, 'testPairs.pkl')
p5 = os.path.join(data_dir, 'testLabels.pkl')
p6 = os.path.join(data_dir, 'wiDict.pkl')
p7 = os.path.join(data_dir, 'iwDict.pkl')
p8 = os.path.join(data_dir, 'commId2Txt.pkl')

p12 = os.path.join(data_dir, 'goldenAnswers.pkl')

p9 = os.path.join(data_dir, 'articleId2Txt.pkl')
p10 = os.path.join(data_dir, 'docIdComm.pkl')
p11 = os.path.join(data_dir, 'numComments.pkl')


def getFilelist(en_dir):
    filelist = []
    for i in range(1,13):
        childDir = os.path.join(en_dir, '{}'.format(i))
    #    print(childDir)
        for r, dirs, files in os.walk(childDir):
            for file in files:
                if file == "{}.txt".format(i):
                    filelist.append(os.path.join(childDir, file))
    return filelist

###############################################################################
def splitAC(file):
    articleID = file.split('/')[-1][:-4]
    wholeArticle = ""
    commentDict = {}
    flag = 'doc'
    with open(file, 'r') as reader:
        lines = reader.readlines()
    reader.close()
    c=0; n=0; docLine=2; commentNo=1
    for lineNum, line in enumerate(lines):
        line = line.strip()
        if line == 'Comments:':
            flag = 'comment'
            article, la = cleanStr(article)
            wholeArticle += '{}'.format(article)
            n += 1
            continue
        
        if flag == 'doc':
            
            try:
                number = int(line.split('.')[0])
                if number == 1:
                    article = line[(len(str(number))+2):]
                    continue
                if number == docLine:                   
                    #article += '\t{}'.format(line[(len(str(number))+2):])
                    article, la = cleanStr(article)
                    wholeArticle += '{}<plp> '.format(article)
                    docLine += 1
                    n += 1
                    article = line[(len(str(number))+2):]
                else:
                    pass
            except:
                pass
            
        elif flag == 'comment':
            
#            if c ==3:
#                raise Exception
            try:
                number = int(line.split('.')[0])
#                print(number)
                
                if number == commentNo:
                    commentID = articleID + "_" + str(number)
                    comment = line[(len(str(number))+2):]
                elif number == commentNo+1:
#                    print('number: {}, commentNo: {}'.format(number, commentNo))
#                    print(comment)
#                    print(commentID)
#                    comment, la = cleanStr(comment)
#                    if la < 1:
#                        commentDict[commentID] = 'nullnull'
#                        c += 1
#                        commentNo += 1
#                    else:
#                        commentDict[commentID] = comment
#                        c += 1
#                        commentNo += 1
                    commentDict[commentID] = comment
                    commId2Txt[commentID] = comment
                    c += 1
                    commentNo += 1                    
                   

                    commentID = articleID + "_" + str(number)
                    comment = line[(len(str(number))+2):]
                else:
                    with open('lostComments.txt', 'a') as w:
                        w.write('article {} lost comment No.{}\n'.format(articleID, commentNo+1))
                    w.close()
                    text = line[(len(str(number))+2):]
                    comment += text
                    
            except:
                comment += ' {}'.format(line)
          
        else:
            raise ValueError
    commentDict[commentID] = comment
    commId2Txt[commentID] = comment  
    c += 1
    print('there are {} sentences and {} comments in article {}'.format(n, c, articleID))
    pk(p8,commId2Txt)
    return articleID, wholeArticle, commentDict

def rewriteYahoo(tFilelist, aFile, cFile, txtMerge):
    with open(aFile, 'w') as aw:
        with open(cFile, 'w') as cw:
            for file in tFilelist:
                articleId, article, commentDict = splitAC(file)
                aw.write('{}\t{}\n'.format(articleId, article.strip()))
                for (commentId, comment) in commentDict.items():
                    comment, la = cleanStr(comment)
                    if la>1:
                        cw.write('{}\t{}\t{}\n'.format(articleId, commentId, comment.strip()))
                    else:
                        ncc = comment.strip() + ' nullnull'
                        cw.write('{}\t{}\t{}\n'.format(articleId, commentId, ncc))
        cw.close()
    aw.close()

def original2txt():
    global commId2Txt
    commId2Txt = {}
    txtFilelist = getFilelist(o_dir)
    rewriteYahoo(txtFilelist, txtArticles, txtComments, txtMerge)

###############################################################################
    
def getDicts():
    aDict = buildDict(d={}, txtFile=txtArticles, tt='ya') ; l1 = len(aDict)
    #aDict = removeFreq(d=aDict, high=4000, low=50) ; l2 = len(aDict)
    aDict = removePercent(d=aDict, p1=0.15, p2=0.15) ; l2 = len(aDict)
    print('aDict from {} words to {} words.'.format(l1, l2))
#    showDictWords(aDict,20)
    cDict = buildDict(d={}, txtFile=txtComments, tt='yc') ; l3 = len(cDict)
    #cDict = removeFreq(d=cDict, high=10000, low=500) ; l4 = len(cDict)
    cDict = removePercent(d=cDict, p1=0.15, p2=0.25) ; l4 = len(cDict)
    print('cDict from {} words to {} words.'.format(l3, l4))
#    showDictWords(cDict,20)
    finalDict = mergeDict(aDict, cDict) ; print('{}+{} ?= {}'.format(l2, l4, len(finalDict)))
    wiDict, iwDict = biDict(finalDict)
#    print('finalDict')
#    showDictWords(finalDict)
    return aDict, cDict, finalDict, wiDict, iwDict

def getMappings(aDict, cDict, finalDict, wiDict, iwDict):
    articleId2Idx = {} ; commId2Idx = {} #; commId2Txt = {}
    trainPairs = [] ; testPairs = [] ; testLabels = []
    articleId2Txt = {} ; docIdComm = {} ; numComments = defaultdict(int)
    #
    with open(txtComments, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            articleID, commentID, comment = line.strip().split('\t')
            #commId2Txt[commentID] = comment
            comm = comment.strip().split(' ') ; commIdx = []
            for word in comm:
                if word in cDict:
                    commIdx.append(wiDict[word])
            if len(commIdx) > 5:
                trainPairs.append((articleID, commentID))
                commId2Idx[commentID] = commIdx    
                if articleID in docIdComm:
                    docIdComm[articleID].append(commentID)
                else:
                    docIdComm[articleID] = [commentID]
                numComments[articleID] += 1
    reader.close()

    with open(txtArticles, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            articleID, article = line.strip().split('\t')
            articleId2Txt[articleID] = article
            article = removePeriod(article)
            atc = article.strip().split(' ') ; atcIdx = []
            for word in atc:
                if word in aDict:
                    atcIdx.append(wiDict[word])
            articleId2Idx[articleID] = atcIdx
    reader.close()

    testPairs = trainPairs
    testLabels = [0 for i in range(len(testPairs))]
    # save
    pk(p1,articleId2Idx)
    pk(p2,commId2Idx)
    pk(p3,trainPairs)
    pk(p4,testPairs)
    pk(p5,testLabels)
    pk(p6,wiDict)
    pk(p7,iwDict)
    #pk(p8,commId2Txt)
    pk(p9,articleId2Txt)
    pk(p10,docIdComm)
    pk(p11,numComments)
    
def mergeTxt():
    with open(txtMerge, 'w') as w:
        with open(txtArticles, 'r') as r1:
            lines = r1.readlines()
            for line in lines:
                line = line.strip().split('\t')[1]
                line = removePeriod(line)
                w.write('{}\n'.format(line))
        r1.close()
        with open(txtComments, 'r') as r2:
            lines = r2.readlines()
            for line in lines:
                line = line.strip().split('\t')[-1]
                w.write('{}\n'.format(line))
        r2.close()
    w.close()

def getPickles():
    aDict, cDict, finalDict, wiDict, iwDict = getDicts()
    getMappings(aDict, cDict, finalDict, wiDict, iwDict)
    mergeTxt()
###############################################################################

def getAnswers():
    goldenAnswers = []
    for i in range(12):
        fpath = os.path.join(o_dir, '{}'.format(i+1))
        file = os.path.join(fpath, 'refalign.txt')
        goldenAnswer = {}
        with open(file, 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                commentIdx, sentIdx = line.strip().split('\t') 
                if commentIdx in goldenAnswer:
                    goldenAnswer[commentIdx].add(sentIdx)
                else:
                    goldenAnswer[commentIdx] = set()
                    goldenAnswer[commentIdx].add(sentIdx)
            goldenAnswers.append(goldenAnswer)
        reader.close()
    pk(p12, goldenAnswers)

def run(args):
    if args.cmd == 'o2t':
        original2txt()
        
    elif args.cmd == 'answer':
        getAnswers()
        
    elif args.cmd == 't2d':
        getPickles()

    elif args.cmd == 'all':
        original2txt()
        getAnswers()
        getPickles()
        
    else:
        pass
    
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cmd', '--cmd', type=str, default='t2d')
    args = parser.parse_args()
    run(args)