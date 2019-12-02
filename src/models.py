# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:52:11 2019

@author: zhouyu
"""

import os

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from random import sample



class VAEencoder(nn.Module):
    def __init__(self, net_arch, dimX, numTopic):
        super(VAEencoder, self).__init__()
        ac = net_arch
        self.numTopic = numTopic
        self.fcEn1 = nn.Linear(dimX, ac.hiddenUnit1)    
        self.fcEn2 = nn.Linear(ac.hiddenUnit1, ac.hiddenUnit2)       
        self.fcMiu = nn.Linear(ac.hiddenUnit2, numTopic)      
        self.meanBN = nn.BatchNorm1d(numTopic)                  
        self.fcSigma = nn.Linear(ac.hiddenUnit2, numTopic)         
        self.sigmaBN = nn.BatchNorm1d(numTopic)      

    def forward(self, x, bn=True):
        en1 = F.softplus(self.fcEn1(x))                       
        en2 = F.softplus(self.fcEn2(en1))  
        if bn:                 
            miu   = self.meanBN(self.fcMiu(en2))       
            sigma = self.sigmaBN(self.fcSigma(en2))   
        else:
            miu   = self.fcMiu(en2)     
            sigma = self.fcSigma(en2)  
        #posterior_var = torch.exp(sigma)
        return miu, sigma
    
    def paras(self):
        return [self.fcEn1, self.fcEn2, self.fcMiu, self.fcSigma]
        
class VAEdecoder(nn.Module):
    def __init__(self, dimZ, dimX):
        super(VAEdecoder, self).__init__()
        self.fcG1 = nn.Linear(dimZ, dimX)
        self.decoder_bn = nn.BatchNorm1d(dimX)
        
    def forward(self, p):
        recon = F.softmax(self.decoder_bn(self.fcG1(p)),dim=1)  
        return recon
    
    def paras(self):
        return [self.fcG1]
    
class OIVAEdecoder(nn.Module):
    def __init__(self, dimZ, dimX):
        super(OIVAEdecoder, self).__init__()
        self.fcM1 = nn.Linear(dimZ, dimZ)
        self.fcM2 = nn.Linear(dimZ, dimZ)
        self.fcG1 = nn.Linear(dimZ, dimX)
        self.fcG2 = nn.Linear(dimZ, dimX)
        self.decoderBN1 = nn.BatchNorm1d(dimX)
        self.decoderBN2 = nn.BatchNorm1d(dimX)
        
    def forward(self, p, bn=True, clf=False):
        za = self.getZa(p)
        zc = self.getZc(p)
        if bn:
            rea = F.softmax(self.decoderBN1(self.fcG1(za)), dim=1)
            rec = F.softmax(self.decoderBN2(self.fcG2(zc)), dim=1)
        else:
            rea = F.softmax(self.fcG1(za), dim=1)
            rec = F.softmax(self.fcG2(zc), dim=1)
        recon = rea+rec  
        if clf:
            return recon, za, zc
        else:
            return recon
    
    def getZa(self, p):
        return self.fcM1(p)
    
    def getZc(self, p):
        return self.fcM2(p)
    
    def paras(self):
        return [self.fcM1, self.fcM2, self.fcG1, self.fcG2]

    
class DISCRIMINATOR(nn.Module):
    def __init__(self, net_arch, dimX, dimH, dimC):
        super(DISCRIMINATOR, self).__init__()
        self.na = net_arch
        self.dimX = dimX
        self.dimC = dimC
        self.fcHidden = nn.Linear(dimX, dimH, bias=True)
        self.fcOut = nn.Linear(dimH, dimC)
        
    def forward(self, x, y, compute_loss=False):
        res = F.relu(self.fcHidden(x))
        res = self.fcOut(res)
        if self.na.gpu:
            res = res.cuda()
            y = y.cuda()
        if compute_loss:
            dloss = self.loss(res,y)
            return res, dloss
        #sm = torch.nn.LogSoftmax(-1)
        else:
            if self.na.gpu:
                return res.cuda()
            else:
                return res
    
    def loss(self, res, y):
        lossF = torch.nn.CrossEntropyLoss()
        loss = lossF(res, y)
        return loss
    
    def paras(self):
        return [self.fcHidden]
    


class standardVAE(nn.Module):
    def __init__(self, net_arch, dimX, numTopic):
        super(standardVAE, self).__init__()
        self.na = net_arch
        self.dimZ = numTopic
        self.encoder = VAEencoder(net_arch=net_arch, dimX=dimX, numTopic=numTopic)
        self.decoder = VAEdecoder(dimZ=numTopic, dimX=dimX)
        self.dropP = nn.Dropout(0.2)
        
        prior_mean   = torch.Tensor(1, numTopic).fill_(0)
        prior_var    = torch.Tensor(1, numTopic).fill_(net_arch.variance)
        prior_logvar = torch.log(prior_var)
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        if net_arch.initMult != 0:
            self.decoder.fcG1.weight.data.uniform_(0, net_arch.initMult)
        
        
    def forward(self, x, y, compute_loss=False, avg_loss=True):
        if compute_loss:
            p, miu, sigma, posterior_var = self.getP(x, compute_loss)
            recon = self.decoder(p)
            nl, kld, loss = self.loss(x, recon, miu, sigma, posterior_var, avg_loss)
            return recon, nl, kld, loss
        else:
            p = self.getP(x, compute_loss)
            recon = self.decoder(p)
            return recon
    
    def getP(self, x, compute_loss):
        miu, sigma = self.encoder(x, compute_loss)
        posterior_var = torch.exp(sigma)
        eps = Variable(torch.randn(1,self.dimZ))
        if self.na.gpu:
            eps = eps.cuda()
        z = miu + posterior_var.sqrt() * eps
        #z = miu + sigma * eps 
        p = F.softmax(z,dim=1) 
        p = self.dropP(p) 
        if compute_loss:
            return p, miu, sigma, posterior_var
        else:
            return p
    
    def loss(self, x, recon, miu, sigma, posterior_var, avg_loss):
        NL = -torch.sum(x.mul(torch.log(recon+1e-10)), dim=1)

        prior_mean   = Variable(self.prior_mean).expand_as(miu)
        prior_var    = Variable(self.prior_var).expand_as(miu)
        prior_logvar = Variable(self.prior_logvar).expand_as(miu)
        var_division    = posterior_var  / prior_var
        diff            = miu - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - sigma
        KLD = 0.5 * (torch.sum((var_division + diff_term + logvar_division),dim=1) - self.dimZ)
        #KLD = 0.5*(torch.sum(( torch.exp(sigma) + miu + sigma ), dim=1) - self.dimZ)
        #KLD = 0.5*torch.sum(1 + torch.log(torch.pow(asigma,2)) - torch.pow(amiu,2) - torch.pow(asigma,2))
        loss = NL + KLD
        if avg_loss:
            return NL.mean(), KLD.mean(), loss.mean()
        else:
            return NL, KLD, loss
    
    def writeParas(self, writer, statement):
        paras = self.encoder.paras() + self.decoder.paras()
        with open(writer, 'a') as w:
            w.write('{}\n'.format(statement))
            for i, para in enumerate(paras):
                w.write('parameter {}:\n{}\n'.format(i, para.weight.data))
        w.close()
  
class OIVAE(nn.Module):
    def __init__(self, net_arch, dimX, numTopic):
        super(OIVAE, self).__init__()
        self.na = net_arch
        self.dimZ = numTopic
        self.encoder = VAEencoder(net_arch=net_arch, dimX=dimX, numTopic=numTopic)
        self.decoder = OIVAEdecoder(dimZ=numTopic, dimX=dimX)
        self.dropP = nn.Dropout(0.2)
        
        
        prior_mean   = torch.Tensor(1, numTopic).fill_(0)
        prior_var    = torch.Tensor(1, numTopic).fill_(net_arch.variance)
        prior_logvar = torch.log(prior_var)
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        if net_arch.initMult != 0:
            self.decoder.fcG1.weight.data.uniform_(0, net_arch.initMult)
        
    def forward(self, x, y, compute_loss=False, avg_loss=True):
        if compute_loss:
            p, miu, sigma, posterior_var = self.getP(x, compute_loss)
            recon = self.decoder(p)
            nl, kld, loss = self.loss(x, recon, miu, sigma, posterior_var, avg_loss)
            return recon, nl, kld, loss
        else:
            p = self.getP(x, compute_loss)
            recon = self.decoder(p)
            return recon
    
    def getP(self, x, compute_loss=False):
        miu, sigma = self.encoder(x, bn=compute_loss)
        posterior_var = torch.exp(sigma)
        eps = Variable(torch.randn(1,self.dimZ))
        if self.na.gpu:
            eps = eps.cuda()
        #z = miu + sigma * eps 
        z = miu + posterior_var.sqrt() * eps
        p = F.softmax(z,dim=1) 
        p = self.dropP(p) 
        if compute_loss:
            return p, miu, sigma, posterior_var
        else:
            return p
    
    def loss(self, x, recon, miu, sigma, posterior_var, avg_loss):
        NL = -torch.sum(x.mul(torch.log(recon+1e-10)), dim=1)

        prior_mean   = Variable(self.prior_mean).expand_as(miu)
        prior_var    = Variable(self.prior_var).expand_as(miu)
        prior_logvar = Variable(self.prior_logvar).expand_as(miu)
        var_division    = posterior_var  / prior_var
        diff            = miu - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - sigma
        KLD = 0.5 * (torch.sum((var_division + diff_term + logvar_division),dim=1) - self.dimZ)
        
        #KLD = 0.5*(torch.sum(( torch.exp(sigma) + miu + sigma ), dim=1) - self.dimZ)
        #KLD = 0.5*torch.sum(1 + torch.log(torch.pow(asigma,2)) - torch.pow(amiu,2) - torch.pow(asigma,2))
        loss = NL + KLD
        if avg_loss:
            return NL.mean(), KLD.mean(), loss.mean()
        else:
            return NL, KLD, loss
    
    def writeParas(self, writer, statement):
        paras = self.encoder.paras() + self.decoder.paras()
        with open(writer, 'a') as w:
            w.write('{}\n'.format(statement))
            for i, para in enumerate(paras):
                w.write('parameter {}:\n{}\n'.format(i, para.weight.data))
        w.close()

class OIVAESOCC(nn.Module):
    def __init__(self, net_arch, dimX, numTopic, clf=False):
        super(OIVAESOCC, self).__init__()
        self.na = net_arch
        self.dimZ = numTopic
        self.clf = False
        self.encoder = VAEencoder(net_arch=net_arch, dimX=dimX, numTopic=numTopic)
        self.decoder = OIVAEdecoder(dimZ=numTopic, dimX=dimX)
        self.dropP = nn.Dropout(0.2)
        if clf:
            self.fcCLF = DISCRIMINATOR(net_arch, dimX=numTopic*2, dimH=20, dimC=2) 
            self.clf = clf
        
        prior_mean   = torch.Tensor(1, numTopic).fill_(0)
        prior_var    = torch.Tensor(1, numTopic).fill_(net_arch.variance)
        prior_logvar = torch.log(prior_var)
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        if net_arch.initMult != 0:
            self.decoder.fcG1.weight.data.uniform_(0, net_arch.initMult)
        
    def forward(self, x, y, compute_loss=False, avg_loss=True):
        if compute_loss:
            p, miu, sigma, posterior_var = self.getP(x, compute_loss)
            if self.clf:
                recon, za, zc = self.decoder(p, bn=avg_loss, clf=self.clf)
                x_clf = torch.cat((za,zc),dim=-1)
                res, clfLoss = self.fcCLF(x_clf,y,compute_loss=self.clf)
            else:
                recon = self.decoder(p, bn=avg_loss, clf=self.clf)
            nl, kld, loss = self.loss(x, recon, miu, sigma, posterior_var, avg_loss)
            if self.clf:
                return recon, nl, kld, loss+clfLoss
            else:
                return recon, nl, kld, loss
        else:
            p = self.getP(x, compute_loss)
            recon = self.decoder(p)
            return recon
    
    def getP(self, x, compute_loss=False):
        miu, sigma = self.encoder(x, bn=compute_loss)
        posterior_var = torch.exp(sigma)
        eps = Variable(torch.randn(1,self.dimZ))
        if self.na.gpu:
            eps = eps.cuda()
        #z = miu + sigma * eps 
        z = miu + posterior_var.sqrt() * eps
        p = F.softmax(z,dim=1) 
        p = self.dropP(p) 
        if compute_loss:
            return p, miu, sigma, posterior_var
        else:
            return p
    
    def loss(self, x, recon, miu, sigma, posterior_var, avg_loss):
        NL = -torch.sum(x.mul(torch.log(recon+1e-10)), dim=1)

        prior_mean   = Variable(self.prior_mean).expand_as(miu)
        prior_var    = Variable(self.prior_var).expand_as(miu)
        prior_logvar = Variable(self.prior_logvar).expand_as(miu)
        var_division    = posterior_var  / prior_var
        diff            = miu - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - sigma
        KLD = 0.5 * (torch.sum((var_division + diff_term + logvar_division),dim=1) - self.dimZ)
        
        #KLD = 0.5*(torch.sum(( torch.exp(sigma) + miu + sigma ), dim=1) - self.dimZ)
        #KLD = 0.5*torch.sum(1 + torch.log(torch.pow(asigma,2)) - torch.pow(amiu,2) - torch.pow(asigma,2))
        loss = NL + KLD
        if avg_loss:
            return NL.mean(), KLD.mean(), loss.mean()
        else:
            return NL, KLD, loss
    
    def writeParas(self, writer, statement):
        paras = self.encoder.paras() + self.decoder.paras()
        if self.clf:
            paras = self.encoder.paras() + self.decoder.paras() + self.fcCLF.paras()
        with open(writer, 'a') as w:
            w.write('{}\n'.format(statement))
            for i, para in enumerate(paras):
                w.write('parameter {}:\n{}\n'.format(i, para.weight.data))
        w.close()
