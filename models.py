import torch
from torch import nn
from torch.autograd import Variable
from visualize import display
from rnn import LSTMCell

from dataloader import get_train_valid_loader, get_test_loader

from Unet import UNet, encoderNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    Encoder: based on pretrained unet encoder part
    """

    def __init__(self, in_channels, depth, start_filters, dict_path, if_fine_tune=False):
        """
        :param in_channels: the channels of input image
        :param depth: the depth of the encoder
        :param start_filters: the number of filters in the first encode block
        :param dict_path: the state_dict of pretrained model
        :param if_fine_tune: whther fine-tune the encoder part
        """
        super(Encoder, self).__init__()
        self.if_fine_tune = if_fine_tune

        # pretrained_model = UNet(in_channels=in_channels, depth=depth, start_filters=start_filters, merge_mode=merge_mode)
        encoder = encoderNet(in_channels=in_channels, depth=depth, start_filters=start_filters)

        pretrained_dict = torch.load(dict_path)
        encoder_dict = encoder.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)
        encoder.load_state_dict(encoder_dict)
        self.encoder = encoder
        self.fine_tune()


    def forward(self, slice):
        """
        Forward propagation
        :param slice: a slice of sMRI data, a tensor of dimensions (batch_size, channel=1, image_size, image_size)
        :return: encoded features (batch_size, image_size//(2**(depth-1)), image_size//(2**(depth-1)), channel=start_filters*(2**(depth-1)))
        """
        out = self.encoder(slice)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self):
        """
        Allow or prevent the computation of gradients for decoder blocks
        :return: None
        """
        for params in self.encoder.parameters():
            params.requires_grad = self.if_fine_tune



class Attention(nn.Module):
    """
    Attention network
    """
    def __init__(self, encoder_channels, rnn_channels, attention_channels):
        """
        :param encoder_channels: feature channels of encoded slice
        :param rnn_channels: channels of rnn
        :param attention_channels: channels of the soft attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_channels, attention_channels) # linear layer to transform encoded image to soft attention size (bacth_size, num_pixels, soft_channels)
        self.rnn_att = nn.Linear(rnn_channels, attention_channels) # linear layer to transform previous hidden state to soft attention size (bacth_size, soft_channels)
        self.full_att = nn.Linear(attention_channels, 1) # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(dim=1) # softmax layer to calculate weights

    def forward(self, encoder_out, prev_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (bacth_size, num_pixels, encoder_channels)
        :param prev_hidden: previous hidden layer of decoder rnn, a tensor of dimension (batch_size, decoder_channels)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out) # (batch_size, num_pixels, attention_dim)
        att2 = self.rnn_att(prev_hidden) # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) # (bacth_size, num_pixels)
        alpha = self.softmax(att) # (batch_size, num_pixels)
        attention_weighted_coding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (batch_size, encoder_channels)
        return attention_weighted_coding, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder: lstm
    """
    def __init__(self, encoder_channels, rnn_channels, attention_channels):
        """
        :param encoder_channels: feature channels of encoded images
        :param rnn_channels: channels of rnn decoder (multilayer: [c1, c2, c2, ...])
        :param attention_channels: channels of attention network
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_channels = encoder_channels
        self.rnn_channels = rnn_channels
        self.attention_channels = attention_channels
        # self.dropout = dropout

        self.input_channels = [encoder_channels] + rnn_channels
        self.hidden_channels = rnn_channels
        self.num_layers = len(self.hidden_channels)
        self.decoder_lstm = []
        # self.init_hidden = []
        # self.init_cell = []
        # self.init_h = nn.Linear(self.encoder_channels[0], self.rnn_channels[0])
        # self.init_c = nn.Linear(self.encoder_channels[0], self.rnn_channels[0])
        self.last_channel = self.hidden_channels[-1]

        # attention part
        self.attention = Attention(encoder_channels=self.encoder_channels, rnn_channels=self.last_channel, attention_channels=self.attention_channels) # attention network

        # decoder part
        for i in range(self.num_layers):
            layer_name = 'layer{}'.format(i)
            layer_cell = LSTMCell(self.input_channels[i], self.hidden_channels[i])
            setattr(self, layer_name, layer_cell)
            self.decoder_lstm.append(layer_cell)

        # init part: linear layer to find initial hidden state and cell state of first LSTMCell
        # for i in range(self.num_layers):
        #     inith_name = 'layer{}_inith'.format(i)
        #     initc_name = 'layer{}_initc'.format(i)
        #     init_h = nn.Linear(self.input_channels[i], self.hidden_channels[i])
        #     init_c = nn.Linear(self.input_channels[i], self.hidden_channels[i])
        #     setattr(self, inith_name, init_h)
        #     setattr(self, initc_name, init_c)
        #     self.init_hidden.append(init_h)
        #     self.init_cell.append(init_c)

        # classification part......

    # def init_hidden_state(self, encoder_out):
    #     """
    #     Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
    #     :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_channels)
    #     :return: hidden_state, cell_state
    #     """
    #     mean_encoder_out = encoder_out.mean(dim=1)
    #     h = self.init_h(mean_encoder_out)
    #     c = self.init_c(mean_encoder_out)
    #     return h, c

    def forward(self, encoder_out, prev_hidden, prev_internal_state):
        """
        Forward propagation.
        :param encoder_out: encoded slice images, a tensor of dimension (batch_size, feature_size, feature_size, encoder_channels)
        :param prev_hidden: previous hidden state, a tensor of dimension (batch_size, rnn_channels)
        :param prev_internal_state: a list of previous internal state.
        :return: step_last_hidden_state, step_classification_probability, step_internal_state
        """
        # batch_size = encoder_out.size(0)
        # encoder_channels = encoder_out.size(-1)

        # flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_channels) # a tensor of dimension (batch_size, num_pixels, encoder_channels)

        # obtain attention weight and attention weighted encoding
        attention_weighted_encoding, alpha = self.attention(encoder_out, prev_hidden)

        # forward propagation of lstm encoder part
        cur_internal_state = []
        cur_input = attention_weighted_encoding
        for i in range(self.num_layers):
            layer_name = 'layer{}'.format(i)
            (prev_h, prev_c) = prev_internal_state[i]
            cur_input, cur_c = getattr(self, layer_name)(cur_input, prev_h, prev_c)
            cur_internal_state.append((cur_input, cur_c))

        return cur_input, cur_c, cur_internal_state, alpha


class AttentionMRI(nn.Module):
    def __init__(self, encoder_depth, encoder_start_filters, encoder_dict_path, decoder_rnn_channels, decoder_attention_channels, number_step, number_class, dropout=0.5):
        super(AttentionMRI, self).__init__()

        self.encoder_depth = encoder_depth
        self.encoder_start_filters = encoder_start_filters
        self.encoder_dict_path = encoder_dict_path
        self.encoder_channels = self.encoder_start_filters * (2**(self.encoder_depth-1))
        self.encoder_scalar = 2**(self.encoder_depth-1)
        self.decoder_rnn_channels = decoder_rnn_channels
        self.decoder_number_layers = len(self.decoder_rnn_channels)
        self.decoder_attention_channels = decoder_attention_channels
        self.number_step = number_step
        self.number_class = number_class
        self.dropout = dropout

        # encoder
        self.encoder = Encoder(in_channels=1, depth=self.encoder_depth, start_filters=self.encoder_start_filters, dict_path=self.encoder_dict_path)

        # decoder with attention
        self.decoder = DecoderWithAttention(encoder_channels=self.encoder_channels, rnn_channels=self.decoder_rnn_channels, attention_channels=self.decoder_attention_channels)

        # linear layers to initialize the hidden state and cell state for lstm
        self.init_h = nn.Linear(self.encoder_channels, self.decoder_rnn_channels[0])
        self.init_c = nn.Linear(self.encoder_channels, self.decoder_rnn_channels[0])

        # classification layer for each step and the last cell state
        self.fc_each_step = nn.Linear(self.decoder_rnn_channels[-1], self.number_class)

        # classification layer for the last cell state
        self.fc_last_cell = nn.Linear(self.decoder_rnn_channels[-1], self.number_class)

        # classification layer for all step
        self.fc_all_step = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.number_step*self.decoder_rnn_channels[-1], 128),
            nn.ReLU(True),
            nn.Linear(128, self.number_class)
        )

        self.softmax = nn.Softmax()

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell state for each layer of the decoder's lstm
        :param encoder_out: encoded images, a tensor of (batch_size, num_pixels*num_step, encoder_channels)
        :return: initial internal state list
        """
        init_internal_state = []
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        init_internal_state.append((h, c))

        for i in range(1, self.decoder_number_layers):
            init_internal_state.append((
                Variable(torch.zeros(1, self.decoder_rnn_channels[i])).to(device),
                Variable(torch.zeros(1, self.decoder_rnn_channels[i])).to(device)
            ))
        return init_internal_state

    def forward(self, mri):
        """
        Forward propagation.
        :param mri: mri data, a tensor of dimension (batch_size, channels, x, y, h)
        :return:
        """
        batch_size, channel, x, y, h = mri.size()

        # encode part
        # create tensor to hold encoded mri data
        encoded_data = torch.zeros(batch_size, self.number_step, (x//self.encoder_scalar)*(h//self.encoder_scalar), self.encoder_channels).to(device)
        for i in range(self.number_step):
            encoded_feature = self.encoder(mri[:, :, :, i, :])
            encoded_feature = encoded_feature.view(batch_size, -1, self.encoder_channels)
            encoded_data[:, i, :, :] = encoded_feature

        # generate the initial internal state of lstm
        encoded_data_reshape = encoded_data.view(batch_size, -1, self.encoder_channels)
        init_internal_state = self.init_hidden_state(encoded_data_reshape)

        # decode part
        # create tensor to the step output
        alpha = torch.zeros(batch_size, self.number_step, (x//self.encoder_scalar)*(h//self.encoder_scalar)).to(device)
        hidden = torch.zeros(batch_size, self.number_step, self.decoder_rnn_channels[-1]).to(device)
        each_step_score = torch.zeros(batch_size, self.number_step, self.number_class).to(device)
        step_internal_state = init_internal_state
        (step_hidden, _) = init_internal_state[-1]
        for i in range(self.number_step):
            step_hidden, step_cell, step_internal_state, step_alpha = self.decoder(encoded_data[:, i, :, :], step_hidden, step_internal_state)
            alpha[:, i, :] = step_alpha
            hidden[:, i, :] = step_hidden
            each_step_score[:, i, :] = self.softmax(self.fc_each_step(step_hidden))

        cell = step_cell

        # classification part
        # 1. for all hidden state
        all_step_score = self.softmax(self.fc_all_step(hidden))
        # 2. for the last cell state
        last_cell_score = self.softmax(self.fc_last_cell(cell))

        return each_step_score, all_step_score, last_cell_score


########################################################################################################################
#
# pretrained_model_path = './unet_3layer_4begin.ckpt'
#
# model = UNet(1, depth=3, start_filters=4, merge_mode='concat')
#
# # print(model)
#
# # for param in model.parameters():
# #     param.requires_grad = False
#
# encoder = encoderNet(1, depth=3, start_filters=4)
#
# #print(encoder)
#
# model.load_state_dict(torch.load(pretrained_model_path))
#
# pretrained_dict = model.state_dict()
#
# encoder_dict = encoder.state_dict()
#
#
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
#
# encoder_dict.update(pretrained_dict)
# encoder.load_state_dict(encoder_dict)
#
# # print(encoder)
#
# # test the pretrained unet
# test_loader = get_test_loader(1)
# test_list = list(test_loader)
#
# test_sample = test_list[20]
#
# test_data = test_sample['data'][:, :, :, 50, :]
#
# test_data = test_data.to(device)
#
# print('model1')
# model.to(device)
# model.eval()
# with torch.no_grad():
#     test_output, test_encode = model(test_data)
#
# # display(test_input)
# # display(test_output)
# # display(test_encode[1])
# # display(test_residual)
#
# print('encoder')
# encoder.to(device)
# encoder.eval()
# with torch.no_grad():
#     encode = encoder(test_data)
# display(encode)
#
# print('ori_encoder')
# ori_encoder = encoderNet(1, depth=3, start_filters=4)
# ori_encoder.to(device)
# ori_encoder.eval()
# with torch.no_grad():
#     encode = ori_encoder(test_data)
# display(encode)
########################################################################################################################