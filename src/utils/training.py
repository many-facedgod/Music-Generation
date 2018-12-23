###############################################################################################
# Utils for training music models.
#
# Example Usage:
# <add here>
###############################################################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# NOTE:
# Consider avoiding song boundaries... noteslist is a concatenation of ALL songs, thus ignoring boundaries -
# might be possible to use ConcatDataset post-batching each into seq_len; however, this might bias toward longer songs 
# (though maybe they're equal enough that it's not problematic).
# Additionally, we might need something smarter to generate good song endings. But let that come later :)
class NotesDataset(Dataset):
    def __init__(self, noteslist):
        self.seq_len =  100
        no_seq = len(noteslist) // self.seq_len
        data = noteslist[:no_seq * self.seq_len]
        self.data=torch.tensor(data).view(-1,self.seq_len)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,index):
        arr = self.data[index]
        x = arr[:-1]
        y = arr[1:]
        return x,y

class MusicModel(nn.Module):
    def __init__(self,vocab_size,embed_size=400,hidden_size=1250,nlayers=2):
        super(MusicModel,self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=nlayers)
        self.fc_embed = nn.Linear(hidden_size,embed_size)
        #weight tying

    def forward(self,x,hidden=None,gen=False):
        # LX N
        batch_size = x.shape[1]  #N
        embed = self.embedding(x)  # L x N x E
        out_LSTM,hidden = self.lstm(embed,hidden)
        out_LSTM=out_LSTM.view(-1,self.hidden_size)  #(L*N)x H
        out = self.fc_embed(out_LSTM)  # (L*N) x E
        out = torch.mm(out,torch.transpose(self.embedding.weight,0,1))  #(L*N)*V
        out = out.view(-1,batch_size,self.vocab_size)  # Lx N X V

        if gen==False:
            return out
        else:
            return out,hidden

def train_batch(model, optimizer, criterion, inputs, labels, device="cpu"):
    inputs = inputs.to(device).long()
    labels = labels.to(device).long()
    output = model(inputs)
    output = output.view(-1, output.shape[2])
    loss = criterion(output, labels.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, optimizer, criterion, train_dataloader, num_epochs=100, device="cpu"):
    model.train().to(device)
    for epoch in range(num_epochs):
        total_epoch_loss = 0
        for batch_no, (x,labels) in enumerate(train_dataloader):
            loss = train_batch(model, optimizer, criterion, x, labels, device)
            total_epoch_loss += loss
            print('Epoch:{}/{} batch_no:{}  Batch Loss:{:.4f}'.format(epoch+1, num_epochs, batch_no, loss))
        epoch_loss = total_epoch_loss / (batch_no+1)
        torch.save(model.state_dict(), "models/checkpoint.pt")
        print('\nEpoch:{}/{} Epoch_Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
        print('------------------------')
