import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 

'''TC-EMNet model'''
PPMI_FEATURE = 80
ADNI_FEATURE = 35
class TC_EMNet(nn.Module):
    def __init__(self, args):
        super(TC_EMNet, self).__init__()
        if args.data_type == 0:
            self.input_size = ADNI_FEATURE
            self.output_size = self.input_size if args.clustering else 3#labels for ADNI dataset
        elif args.data_type == 1:
            self.input_size = PPMI_FEATURE
            self.output_size = self.input_size if args.clustering else 6 #labels for PPMI dataset
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.rnn_encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, dropout=0, bidirectional=False, num_layers=args.num_layers)
        if args.use_label:
            self.label_embed = args.label_embed
           
        self.out =  nn.Sequential(  nn.Dropout(p=args.dropout),
                                    nn.Linear(in_features=self.hidden_size * 2 , out_features=self.output_size))
        self.hidden_mean = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.Sigmoid()
        )
        self.hidden_std = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU()
        )
        self.memory_write = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )
        self.memory_out = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=self.hidden_size, out_features=self.input_size)
        )
        if args.use_label:
            self.target_embed = nn.Sequential(
                nn.Linear(in_features=self.output_size, out_features=self.label_embed),
            )
            self.patient_write = nn.Sequential(
                nn.Linear(in_features=self.label_embed, out_features=self.label_embed),
                nn.ReLU(),
            )
            self.afm = nn.Sequential(
                nn.Linear(in_features=self.label_embed, out_features=self.hidden_size),
                nn.Sigmoid()
            )
       
        self.all_mean = list()
        self.all_std = list()
        self.args = args

    def reparametrize(self, mu, std):
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def memory_network(self, hidden):
        activation = nn.Softmax(dim=1)
        seq_len = hidden.size(1) #batch x seq x hidden
        #print(hidden.size())
        memory = self.memory_write(hidden) #batch x seq (M) x hidden
        output = torch.tensor([], device=self.args.device)
        for i in range(seq_len):
            curr = hidden[:, i, ...] #batch x hidden
            dot = activation(torch.matmul(memory, curr.unsqueeze(-1))) #batch x M x 1
            #print(dot.view(hidden.size(0), 1, -1))
            weights = dot.cpu().detach().numpy()
            np.save(f'{self.args.output_dir}/weights.npy', weights)
            att = memory * dot #batch x M x hidden
            out = torch.sum(att, axis=1).unsqueeze(1) #batch x 1 x hidden (@ curr)
            output = torch.cat([output, out], dim=1)
        return output #batch x seq x hidden

    def memory_network_target(self, hidden):
        activation = nn.Softmax(dim=1)
        seq_len = hidden.size(1) #batch x seq x hidden
        #print(hidden.size())
        hidden = self.target_embed(hidden)
        memory = self.patient_write(hidden) #batch x seq (M) x hidden
        #memory = hidden
        output = torch.tensor([], device=self.args.device)
        for i in range(seq_len):
            curr = hidden[:, i, ...] #batch x hidden
            dot = activation(torch.matmul(memory[..., :(i+1), :], curr.unsqueeze(-1))) #batch x M x 1
            #print(dot.view(hidden.size(0), 1, -1))
            weights = dot.cpu().detach().numpy()
            np.save(f'{self.args.output_dir}/weights.npy', weights)
            att = memory[..., :(i+1), :] * dot #batch x M x hidden
            out = torch.sum(att, axis=1).unsqueeze(1) #batch x 1 x hidden (@ curr)
            output = torch.cat([output, out], dim=1)
        return output #batch x seq x hidden

    def _afm(self, memory, target):
        seq_len = memory.size(1) 
        output = torch.tensor([], device=self.args.device)
        #local_h = self.nn_conv(local_h).squeeze(-1)
        mapping = self.afm(target)
        for i in range(seq_len):
            curr = memory[:, i, ...] #batch x 1 x hidden
            out = curr * mapping[:, i, ...]
            output = torch.cat([output, out.unsqueeze(1)], dim=1)
        return output #batch x seq x hidden

    def forward(self, input, label=None):
        #current diagnosis only based on past visits
        target = torch.zeros(size=label.size(), device=self.args.device)
        target[..., 1:, :] = label[..., :-1, :]

        output, _ = self.rnn_encoder(input)

        memory_out = self.memory_network(output)
        if self.args.use_label:
            memory_target = self.memory_network_target(target)
        for i in range(self.args.hop):
            memory_out = self.memory_network(memory_out)
            if self.args.use_label:
                memory_target = self.memory_network_target(memory_target)
        if self.args.use_label:
            memory_out = self._afm(memory_out, memory_target)
        mean = self.hidden_mean(output)
        std = self.hidden_std(output)

        mu = mean.cpu().detach().numpy()
        sigma = std.cpu().detach().numpy()
        np.save(f'{self.args.output_dir}/mean.npy', mu)
        np.save(f'{self.args.output_dir}/std.npy', sigma)
        
        z = self.reparametrize(mean, std)

        hidden = z.cpu().detach().numpy()
        np.save(f'{self.args.output_dir}/z.npy', hidden)

        z = torch.cat([z, memory_out], dim=2)

        final = z.cpu().detach().numpy()
        np.save(f'{self.args.output_dir}/z_final.npy', final)
        x = self.out(z)
       
        return x, z, mean, std, self.memory_out(memory_out)