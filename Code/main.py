import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Seq2Seq
from train_test import train, test, validation
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset, create_dictionaries
from torch.optim import lr_scheduler
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
param = {
    'dataPath': '../Data/',
    'resume': True,
    'resume_from': 38,
    'checkPointPath': '../Checkpoints/',
    'hidden_dim': 256,
    'value_size': 128,
    'key_size': 128,
    'lr': 0.001,
    'nepochs': 5
}
def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=param['hidden_dim'], 
    value_size=param['value_size'], key_size=param['key_size'], isAttended=True)
    optimizer = optim.Adam(model.parameters(), lr=param['lr'])

    ## Load from pretrained
    if(param['resume'] == True):
        checkPointPath = param['checkPointPath'] + '/epoch' + str(param['resume_from'])
        checkpoint = torch.load(checkPointPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

    scheduler = lr_scheduler.StepLR(optimizer, step_size = 6, gamma = 0.95)
    criterion = nn.CrossEntropyLoss(ignore_index = 0).to(DEVICE)
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data(param['dataPath'])
    print("finished loading data")
    letter2index, index2letter = create_dictionaries(LETTER_LIST)
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)
    print("finished transforming data")

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    print("start training")
    start_epoch = param['resume_from']
    dist = validation(model, val_loader, index2letter)
    test(model, test_loader, index2letter)
    for epoch in range(start_epoch, start_epoch + param['nepochs']):
        train(model, train_loader, val_loader, criterion, optimizer, epoch)
        path = param['checkPointPath'] + "/epoch" + str(epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)
        # val()
        scheduler.step()
        dist = validation(model, val_loader, index2letter)
        print("validation dist is: ", dist)
    test(model, test_loader, index2letter)


if __name__ == '__main__':
    main()