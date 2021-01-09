import time
import torch
import Levenshtein as ls
from dataloader import create_dictionaries
import csv
### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
letter2index, index2letter = create_dictionaries(LETTER_LIST)

def write_csv(fname, prediction):
    rows = []
    for i in range(len(prediction)):
        rows.append([i,  prediction[i]])
    fields = ['id', 'label']
    with open(fname, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

def train(model, train_loader, criterion, optimizer, epoch):
    model.to(DEVICE)
    start = time.time()
    iteration = 1
    total_iteration = len(train_loader)
    model.train()

    for batch_idx, (train, train_len, label, label_len) in enumerate(train_loader):
        optimizer.zero_grad()
        train = train.to(DEVICE)
        label = label.to(DEVICE)
        train_len = train_len.to(DEVICE)
        label_len = label_len.to(DEVICE)
        out = model(epoch, batch_idx, train, train_len, label, isTrain = True)
        loss = criterion(out.permute(0,2,1), label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if(iteration % 5 == 0):
            train_prediction = []
            train_label = []
            total_dist = 0
            num_sample = 0
            logits = out.argmax(dim = -1).to("cpu").numpy()
            label = label.to("cpu").numpy()
            for i in range(logits.shape[0]):
                prediction_str = ''
                label_str = ''
                for idx in logits[i,:]:
                    if(index2letter[idx] == '<eos>' or index2letter[idx] == '<sos>' or index2letter[idx] == '<pad>'):
                        break
                    else:
                        prediction_str += index2letter[idx]
                for idx in label[i,:]:
                    if(index2letter[idx] != '<eos>' and index2letter[idx] != '<sos>'):
                        label_str += index2letter[idx]
                    else:
                        break
                train_prediction.append(prediction_str)
                train_label.append(label_str)
                dist = ls.distance(label_str, prediction_str)
                num_sample += 1
                total_dist += dist
            avr_dist = total_dist/num_sample

            train_filename = "../Results/train_prediction.csv"
            label_filename = "../Results/train_label.csv"
            write_csv(train_filename, train_prediction)
            write_csv(label_filename, train_label)

            print("epoch: {}, iteration: {}/{}, loss is: {}, average distance is: {}".format(epoch, batch_idx, total_iteration, loss.item(), avr_dist))
        
        iteration += 1
        torch.cuda.empty_cache()
    end = time.time()

def validation(model, val_loader, index2letter):
    model.eval()
    model.to(DEVICE)
    iteration = 0
    total_iteration = len(val_loader)
    total_dist = 0
    num_sample = 0
    val_prediction = []
    val_label = []
    num_sample = 0
    for batch_idx, (val, val_len, label, label_len) in enumerate(val_loader):
        val = val.to(DEVICE)
        val_len = val_len.to(DEVICE)
        label = label.to("cpu").numpy()
        out = model(0, batch_idx, val, val_len, isTrain=False)
        logits = out.argmax(dim = -1).to("cpu").numpy()
        for i in range(logits.shape[0]):
            prediction_str = ''
            label_str = ''
            for idx in logits[i,:]:
                if(index2letter[idx] == '<eos>' or index2letter[idx] == '<sos>' or index2letter[idx] == '<pad>'):
                    break
                else:
                    prediction_str += index2letter[idx]
            for idx in label[i,:]:
                if(index2letter[idx] != '<eos>' and index2letter[idx] != '<sos>'):
                    label_str += index2letter[idx]
                else:
                    break
            val_prediction.append(prediction_str)
            val_label.append(label_str)
            dist = ls.distance(label_str, prediction_str)
            num_sample += 1
            total_dist += dist
    avr_dist = total_dist/num_sample
    val_filename = "../Results/val_prediction.csv"
    label_filename = "../Results/val_label.csv"
    write_csv(val_filename, val_prediction)
    write_csv(label_filename, val_label)
    torch.cuda.empty_cache()
    return avr_dist

def test(model, test_loader, index2letter):
    model.eval()
    model.to(DEVICE)
    iteration = 0
    total_iteration = len(test_loader)
    prediction = []
    for batch_idx, (test, test_len) in enumerate(test_loader):
        test = test.to(DEVICE)
        test_len = test_len.to(DEVICE)
        out = model(0, batch_idx, test, test_len, isTrain=False)
        logits = out.argmax(dim = -1).to('cpu').numpy()
        for i in range(logits.shape[0]):
            prediction_str = ''
            for idx in logits[i,:]:
                if(index2letter[idx] != '<eos>' and index2letter[idx] != '<sos>'):
                    prediction_str += index2letter[idx]
                else:
                    break
            prediction.append(prediction_str)
        if(iteration %20 == 0):
            print("iteration: {}/{}".format(iteration, total_iteration))
        iteration += 1

    test_filename = "../Results/prediction.csv"
    write_csv(test_filename, prediction)
