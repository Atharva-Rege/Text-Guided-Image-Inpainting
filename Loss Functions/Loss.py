import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init
import torch.nn.functional as F
from PIL import Image
import random
import math
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torchmetrics.functional import structural_similarity_index_measure
from torch.nn.utils import spectral_norm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from Models import *



### 1. Text-Guided Attention Loss
def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.clone().view(batch_size, -1, ih, iw)



### 2. WGAN LOSS
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates_global, d_interpolates_local = discriminator(interpolates, interpolates)
    
    # Get gradient w.r.t. interpolates
    gradients_global = torch.autograd.grad(
        outputs=d_interpolates_global,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates_global).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
   
    gradients_local = torch.autograd.grad(
        outputs=d_interpolates_local,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates_local).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients_global = gradients_global.view(gradients_global.size(0), -1)
    gradients_local = gradients_local.view(gradients_local.size(0), -1)
    
    gradient_penalty_global = ((gradients_global.norm(2, dim=1) - 1) ** 2).mean()
    gradient_penalty_local = ((gradients_local.norm(2, dim=1) - 1) ** 2).mean()
    
    return (gradient_penalty_global + gradient_penalty_local) / 2

def calc_wgan_loss(discriminator, output, target):
    y_pred_fake_global, y_pred_fake_local = discriminator(output, target)
    y_pred_real_global, y_pred_real_local = discriminator(target, output)
    
    # WGAN loss
    g_loss = -torch.mean(y_pred_fake_global + y_pred_fake_local) / 2
    d_loss = (-torch.mean(y_pred_real_global + y_pred_real_local) + torch.mean(y_pred_fake_global + y_pred_fake_local)) / 2
    
    return g_loss, d_loss



### 3. Text-Image Matching Loss
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)

    res = (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    return res

def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, use_cuda=True, eps=1e-8):
    # ### Mask mis-match samples ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).cpu().numpy().astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        masks = torch.ByteTensor(masks)
        if use_cuda:
            masks = masks.cuda()

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * 10.0

    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0 = scores0.masked_fill(masks.bool(), -float('inf'))  # Removed in-place operation
    scores1 = scores0.transpose(0, 1)
    
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1

def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size, use_cuda=True):
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).byte()
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        
        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        word = word.repeat(batch_size, 1, 1)
        context = img_features
        
        weiContext, attn = func_attention(word, context, 5.0)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        
        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        # Removed in-place operations here
        row_sim = row_sim * 5.0
        row_sim = row_sim.exp()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = torch.cat(masks, 0)
        masks = masks.byte()
        if use_cuda:
            masks = masks.cuda()
    
    similarities = similarities * 10.0
    if class_ids is not None:
        similarities = similarities.masked_fill(masks.bool(), -float('inf'))  # Removed in-place operation
    similarities1 = similarities.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps



### 4. Reconstruction Loss
def get_recon_loss(mask, imgs, result, L1Loss): #, rate_for_mask_loss=0.8
    
    # Ensure mask is a binary mask (0 and 1 values)
    
    unmasked_area = imgs * (1. - mask)   # Isolating the unmasked region of the original image
    masked_area = imgs - unmasked_area   # Isolating the masked region of the original image
    
    unmasked_result = result * (1. - mask)  # Isolating the unmasked region of the result image
    masked_result = result - unmasked_result  # Isolating the masked region of the result image
    
    # Compute L1 loss for both the unmasked and masked regions
    # unmasked_loss = L1Loss(unmasked_result, unmasked_area)  # L1 loss for unmasked region
    masked_loss = L1Loss(masked_result, masked_area)  # L1 loss for masked region

    return masked_loss
    
    # Total loss is the weighted sum of the unmasked and masked losses
    # tot_loss = (unmasked_loss * (1 - rate_for_mask_loss)) + (masked_loss * rate_for_mask_loss)
    
    # return tot_loss