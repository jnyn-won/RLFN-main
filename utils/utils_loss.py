import torch


def select_loss(loss_func):
    if loss_func == 'L1':
        return torch.nn.L1Loss
    if loss_func == 'MSE':
        return torch.nn.MSELoss
    if loss_func == 'CrossEntropy':
        return torch.nn.CrossEntropyLoss
