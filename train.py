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
            for param_group in optimizer.param_groups:
                print('Current learning rate: ' + str(param_group['lr']))
            model.train()

            correctInTrainEpoch = 0
            top5InTrainEpoch = 0
            for batch_num, (inputs, labels) in enumerate(train_loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                with torch.no_grad():
                    _, prediction = torch.max(outputs, dim=1)
                    numCorrect = prediction == labels
                    correctInTrainEpoch += sum(numCorrect)
                    top5 = torch.topk(outputs, 5, dim=1)
                    top5 = top5[:][1]
                    accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
                    top5InTrainEpoch += sum(accuracy)

                if batch_num % output_period == 0:
                    print('[%d:%.2f] loss: %.3f' % (
                        epoch, batch_num*1.0/num_train_batches,
                        running_loss/output_period
                        ))
                    running_loss = 0.0
                    gc.collect()
                # if batch_num > 5:
                    # break

            gc.collect()
            # save after every epoch
            torch.save(model.state_dict(), "models/model.%d" % epoch)

            # TODO: Calculate classification error and Top-5 Error
            # on training and validation datasets here

            # print(outputs)
            # _, prediction = torch.max(outputs, dim=1)
            # print('predicted class: ', prediction)
            # numCorrect = prediction == labels
            # print('classification error: ', sum(numCorrect)/len(labels))
            # top5 = torch.topk(outputs, 5, dim=1)
            # print('Top 5 classes were: ', top5[:][1])
            # top5 = top5[:][1]
            # # print(labels.repeat(1,5).view(5,-1))
            # print(labels.transpose(0,-1))
            # accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
            # print('top5 accuracy: ', sum(accuracy)/len(labels))



            correctInValEpoch = 0
            top5InValEpoch = 0
            model.eval()
            with torch.no_grad():
                for batch_num, (inputs, labels) in enumerate(val_loader, 1):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, prediction = torch.max(outputs, dim=1)
                    numCorrect = prediction == labels
                    correctInValEpoch += sum(numCorrect)
                    top5 = torch.topk(outputs, 5, dim=1)
                    top5 = top5[:][1]
                    accuracy = [1 if int(labels[ind]) in x else 0 for ind, x in enumerate(top5.numpy())]
                    top5InValEpoch += sum(accuracy)
                    # if batch_num > 5:
                        # break

            accuractyString = 'Epoch %d: T1  %.2f, T5 %.2f, V1 %.2f, V5 %.2f' % (
            epoch,
            100*correctInTrainEpoch/batch_size,
            100*top5InTrainEpoch/batch_size,
            100*correctInValEpoch/batch_size,
            100*top5InValEpoch/batch_size,
            )

            print(accuractyString)

            outptfile.write(accuractyString)
            outptfile.write("\n")
                    


            gc.collect()
            epoch += 1

print('Starting training')
run()
print('Training terminated')
