import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn_kd(student_outputs, labels, teacher_outputs, kd_params, criterion):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha     """

    alpha = kd_params.ALPHA
    T = kd_params.TEMPERATURE

    soft_targets = F.softmax(teacher_outputs / T, dim=-1)
    soft_prob = F.softmax(student_outputs / T, dim=-1)
    #soft_prob = F.log_softmax(student_outputs / T, dim=-1)

    kd_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

    student_loss = criterion(student_outputs, labels)
    
    total_loss = alpha * kd_loss + (1. - alpha) * student_loss

    return kd_loss, student_loss, total_loss

'''
def loss_fn_kd(student_outputs, labels, teacher_outputs, kd_params, criterion):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha     """

    alpha = kd_params.ALPHA
    T = kd_params.TEMPERATURE

    kd_loss = F.kl_div(F.log_softmax(student_outputs / T, dim=1), F.log_softmax(teacher_outputs / T, dim=1), reduction='sum', log_target=True) * (T * T) / student_outputs.numel()
    
    student_loss = criterion(student_outputs, labels)
    
    total_loss = alpha * kd_loss + (1. - alpha) * student_loss

    return kd_loss, student_loss, total_loss
'''

'''
def loss_fn_kd(student_outputs, labels, teacher_outputs, kd_params, criterion):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha     """

    alpha = kd_params.ALPHA
    T = kd_params.TEMPERATURE
    
    kd_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (T * T)
              
    student_loss = criterion(student_outputs, labels) 

    total_loss = alpha * kd_loss + (1. - alpha) * student_loss

    return kd_loss, student_loss, total_loss
'''
