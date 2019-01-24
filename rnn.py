import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import math

class LSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # self.out_channels = out_channels
        # self.if_classify = if_classify

        self.gate = nn.Linear(in_channels+hidden_channels, hidden_channels)
        # self.output = nn.Linear(hidden_channels, out_channels)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden_cell(self):
        return (Variable(torch.zeros(1, self.hidden_channels)).cuda(),
                Variable(torch.zeros(1, self.hidden_channels)).cuda())

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)

        # forget gate
        f_gate = self.gate(combined)
        f_gate = self.sigmoid(f_gate)

        # input gate
        i_gate = self.gate(combined)
        i_gate = self.tanh(i_gate)

        # cell gate
        c_gate = self.gate(combined)
        c_gate = self.tanh(c_gate)

        # output gate
        o_gate = self.gate(combined)
        o_gate = self.sigmoid(o_gate)

        # update cell
        new_cell = torch.add(torch.mul(cell, f_gate), torch.mul(i_gate, c_gate))
        new_hidden = torch.mul(self.tanh(new_cell), o_gate)

        # output for classification at the step
        # class_prob_vector = None
        # if self.if_classify:
        #     output = self.output(new_hidden)
            # class_prob_vector = self.softmax(output)

        return new_hidden, new_cell #, class_prob_vector

#####  TEST   #########
# def main1():
#     # gradient check
#     model = gmLSTMmodel1(input_channels=256, hidden_channels=[128, 64, 32], kernel_size=3, num_slice=200, num_class=2).cuda()
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     input = Variable(torch.rand(5,1,200,256,256)).cuda()
#     target = Variable(torch.from_numpy(np.asarray((1, 0, 0, 1, 1),dtype=np.longlong))).cuda()
#
#     output = model(input)
#     res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-2, atol=1e-2, raise_exception=True)
#     print(res)
#
# if __name__ == '__main__':
#     main1()