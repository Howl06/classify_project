from train_lazy_experiment import one_experiment

import torch
import torch.optim as optim
import torch.nn as nn
# model
from model.model_v3 import mobilenet_v3_large
# Focal loss
from utils.focal_loss import FocalLoss
# other optimzer
from optimizer.optimizer_lion import Lion
from optimizer.Ranger22 import Ranger22
from optimizer.ranger21 import Ranger21
# mixup
import torch.nn.functional as F
# add weights
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, weight:torch.Tensor):
        super(SoftTargetCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1) * self.weight * len(self.weight) / self.weight.sum(), dim=-1)
        return loss.mean()

def main():
    hyper_params = {
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 16,
    "num_classes": 27,
    "alpha": 1,# filters ratio
    "input_size": 224,
    "mobilenet_name": mobilenet_v3_large,
    "loss_function_name": nn.CrossEntropyLoss,#FocalLoss, nn.CrossEntropyLoss, SoftTargetCrossEntropy(mixup)
    "optimizer_name": optim.Adam,#optim.Adam,
    "ImbalancedDatasetSampler": False, # ImbalancedDatasetSampler or class_weights
    "class_weights": False,# loss class weights 
    "learning_speed": "no_speed",# constant, log, ln, root, lr++< 98% val acc <lr--
    "early_stop_patient": 10,# use val acc
    "accumulation_steps": 1,# accumulation_steps nouse
    "schedule": None,#optim.lr_scheduler.CosineAnnealingLR,
    "freeze": False,# freeze feature layer weights 
    "optimizer_name_o": "Adam",
    "loss_name": "nn.CrossEntropyLoss",
    "use_ema": "no_ema",# "ema" open ema else close
    "model_ema_decay": 0.995,
    "mixup_cutmix":"no_mix",# data augmentation "mix" use, other no
    }
    for i in range(5):
        one_experiment(hyper_params)

if __name__ == '__main__':
    main()