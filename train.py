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
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO:ptimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.SGD(model.parameters(), lr=1e-3) # we can to stochastic graddescent, adagrad, adadelta,
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)

    epoch = 1
    with open(cwd + "/resnet_18-" + datestr + ".txt", "w") as outptfile:

        while epoch <= num_epochs:
            running_loss = 0.0
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            correctInTrainEpoch = 0
            top5InTrainEpoch = 0
            epoch_samples = 0

            for param_group in optimizer.param_groups:
                print('Current learning rate: ' + str(param_group['lr']))
            model.train()

            for batch_num, (inputs, labels) in enumerate(train_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                # with torch.no_grad():
                #     _, prediction = torch.max(outputs, dim=1)
                #     prediction = prediction.cpu()
                #     outputs = outputs.cpu()
                #     labels = labels.cpu()
                #     numCorrect = prediction.numpy() == labels.numpy()
                #     correctInTrainEpoch += sum(numCorrect)
                #     top5 = torch.topk(outputs, 5, dim=1)
                #     top5 = top5[:][1]
                #     accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
                #     top5InTrainEpoch += sum(accuracy)

                # if batch_num % output_period == 0:
                #     print('[%d:%.2f] loss: %.3f' % (
                #         epoch, batch_num*1.0/num_train_batches,
                #         running_loss/output_period
                #         ))
                #     epoch_train_loss += running_loss
                #     running_loss = 0.0
                #     gc.collect()
                # if batch_num > 5:
                #     print('[%d:%.2f] loss: %.3f' % (
                #         epoch, batch_num*1.0/num_train_batches,
                #         running_loss/output_period
                #         ))
                #     epoch_train_loss += running_loss
                #     running_loss = 0.0
                #     gc.collect()
                #     print(outputs)
                #     _, prediction = torch.max(outputs, dim=1)
                #     print('predicted class: ', prediction)
                #     print('actual class: ', labels)
                #     numCorrect = prediction == labels
                #     print('classification error: ', sum(numCorrect)/len(labels))
                #     top5 = torch.topk(outputs, 5, dim=1)
                #     print('Top 5 classes were: ', top5[:][1])
                #     top5 = top5[:][1]
                #     # print(labels.repeat(1,5).view(5,-1))
                #     print(labels.transpose(0,-1))
                #     accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
                #     print('top5 accuracy: ', sum(accuracy)/len(labels))
                #     break
                acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
                n = outputs.size(0)
                correctInTrainEpoch += acc1[0]*n
                top5InTrainEpoch += acc5[0]*n
                epoch_samples += n
                if batch_num % output_period == 0:
                    print('[%d:%.2f] loss: %.3f' % (
                        epoch, batch_num*1.0/num_train_batches,
                        running_loss/output_period
                        ))
                    epoch_train_loss += running_loss
                    running_loss = 0.0
                    gc.collect()
                
                # if batch_num > 5:
                #     break

            gc.collect()
            # save after every epoch
            torch.save(model.state_dict(), "models/model.%d" % epoch)

            # TODO: Calculate classification error and Top-5 Error
            # on training and validation datasets here




            correctInValEpoch = 0
            top5InValEpoch = 0

            model.eval()
            with torch.no_grad():
                for batch_num, (inputs, labels) in enumerate(val_loader, 1):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
                    n = outputs.size(0)

                    correctInValEpoch += acc1[0]*n
                    top5InValEpoch += acc5[0]*n
                    epoch_val_loss += loss.item()

                    # if batch_num > 5:
                    #     break
                    

                    # _, prediction = torch.max(outputs, dim=1)
                    # prediction = prediction.cpu()
                    # outputs = outputs.cpu()
                    # labels = labels.cpu()
                    # numCorrect = prediction.numpy() == labels.numpy()
                    # correctInValEpoch += sum(numCorrect)
                    # top5 = torch.topk(outputs, 5, dim=1)
                    # top5 = top5[:][1]
                    # accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
                    # top5InValEpoch += sum(accuracy)
                    # if batch_num > 5:
                    #     print('[%d:%.2f] loss: %.3f' % (
                    #     epoch, batch_num*1.0/num_train_batches,
                    #     running_loss/output_period
                    #     ))
                    #     epoch_train_loss += running_loss
                    #     running_loss = 0.0
                    #     gc.collect()
                    #     print(outputs)
                    #     _, prediction = torch.max(outputs, dim=1)
                    #     print('predicted class: ', prediction)
                    #     print('actual class: ', labels)
                    #     numCorrect = prediction == labels
                    #     print('classification error: ', sum(numCorrect)/len(labels))
                    #     top5 = torch.topk(outputs, 5, dim=1)
                    #     print('Top 5 classes were: ', top5[:][1])
                    #     top5 = top5[:][1]
                    #     # print(labels.repeat(1,5).view(5,-1))
                    #     print(labels.transpose(0,-1))
                    #     accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
                    #     print('top5 accuracy: ', sum(accuracy)/len(labels))
                    #     break

            
            accuractyString = 'Epoch %d Train: T1  %.2f, T5 %.2f, Loss %.2f \nEpoch %d Val: V1 %.2f, V5 %.2f, Loss %.2f\n' % (
            epoch,
            correctInTrainEpoch/(epoch_samples),
            top5InTrainEpoch/epoch_samples,
            epoch_train_loss/(epoch_samples),
            epoch,
            correctInValEpoch/(epoch_samples),
            top5InValEpoch/(epoch_samples),
            epoch_val_loss/(epoch_samples),
            )

            print(accuractyString)

            outptfile.write(accuractyString)
            outptfile.write("\n")
                    


            gc.collect()
            epoch += 1

print('Starting training')
run()
print('Training terminated')
