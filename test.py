import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True


from datetime import datetime
import random
import os
import time

import dataset
from models.AlexNet import *
from models.ResNet import *

freshcount = 0
d = datetime.now()
# datestr = d.strftime("%d-%m-%y")
datestr = d.time().strftime("%H:%M:%S")

cwd = os.getcwd() 


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        # now has output we want to print
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def run(model_file):
    # Parameters
    num_epochs = 10
    output_period = 20
    batch_size = 70

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    val_loader, test_loader = dataset.get_val_test_loaders(batch_size)
    num_test_batches = len(test_loader)

    with open(model_file + "_test_out.txt", "w") as outptfile:
        model.eval()

        correctInValEpoch = 0
        top5InValEpoch = 0
        epoch_val_samples = 0
        with torch.no_grad():
            for batch_num, (inputs, labels) in enumerate(val_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
                n = outputs.size(0)

                correctInValEpoch += acc1[0]*n
                top5InValEpoch += acc5[0]*n
                epoch_val_samples += n
            print("top 5 accuracy: ",100-top5InValEpoch.item()/(epoch_val_samples))

        outs = []
        with torch.no_grad():
            for batch_num, (inputs, labels) in enumerate(test_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs).topk(5, 1, True, True)[1]
                outs.extend(list(outputs))
        for name, preds in zip(sorted(os.listdir("data/test/999")), outs):
            outptfile.write("test/{} {}".format(name, " ".join([str(x.item()) for x in list(preds)])))
            outptfile.write("\n")
            

    gc.collect()

print('Starting Validation and Test output')
run(sys.argv[1])
print('Finished')
