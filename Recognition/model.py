import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''
Base On: What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis
'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channel=3, output_channel=512, block=BasicBlock, layers=(1, 2, 5, 3)):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                               1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


'''
Ours Adding
'''


class ClassificationHeader(nn.Module):
    def __init__(self, charset_size, input_embedding_size=512):
        super(ClassificationHeader, self).__init__()
        self.charset_size = charset_size
        self.input_embedding_size = input_embedding_size
        self.header = nn.Sequential(
            nn.Linear(input_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, charset_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        out = self.header(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = ResNet(in_channels)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, word_image):
        embedding = self.feature_extractor(word_image)[:, :, 0, :].permute((0, 2, 1))
        embedding = self.fc(embedding)
        return embedding


class multiheadSelfAttentionSequential(nn.Sequential):
    def forward(self, inputs):
        for module in self._modules.values():
            if 'multiheadattention' in str(module).lower():
                inputs, _ = module(inputs, inputs, inputs)
            else:
                inputs = module(inputs)
        return inputs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, positional_enconding_type='const', init_pos_as_const=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_enconding_type = positional_enconding_type
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe_const = torch.zeros(1, max_len, d_model)
        pe_const[0, :, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe_const[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe_const[0, :, 1::2] = torch.cos(position * div_term[:-1])
        if positional_enconding_type == 'const':
            pe = pe_const
        else:
            pe = torch.arange(max_len).expand((1, max_len)).view((1, max_len))
            self.pos = nn.Embedding(max_len, d_model)
            if init_pos_as_const:
                self.pos.weight = nn.Parameter(pe_const[0, :, :])
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.positional_enconding_type == 'const':
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pos(self.pe[:, :x.size(1)])
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, encoder_type='bilstm', feature_size=256, hidden_size=256, num_layers=4, num_heads_self_attention=4, dropout_self_attention=0.0,
                 positional_enconding_type='const', max_word_length=26, init_pos_as_const=True):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.positional_enconding_type = positional_enconding_type
        if encoder_type == 'bilstm':
            self.encoder = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        elif encoder_type == 'transformer':
            self.pos = PositionalEncoding(hidden_size, dropout_self_attention, max_len=max_word_length, positional_enconding_type=positional_enconding_type, init_pos_as_const=init_pos_as_const)
            self.pre_encoder = nn.Sequential(*[nn.Linear(feature_size, hidden_size), nn.ReLU(), self.pos])
            self.encoder = [nn.TransformerEncoderLayer(hidden_size, num_heads_self_attention, dropout=dropout_self_attention, batch_first=True)] * num_layers
            self.encoder = nn.Sequential(*self.encoder)

    def forward(self, features):
        if self.encoder_type == 'bilstm':
            enc, _ = self.encoder(features)
        elif self.encoder_type == 'transformer':
            pre_enc = self.pre_encoder(features)
            enc = self.encoder(pre_enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, charset_size, input_embedding_size, decoder_type="fc", num_layers=4, num_heads_self_attention=4, dropout_self_attention=0.0,
                 positional_enconding_type='const', max_word_length=26, init_pos_as_const=True):
        super(Decoder, self).__init__()
        self.charset_size = charset_size
        self.positional_enconding_type = positional_enconding_type
        self.decoder_type = decoder_type
        self.header = ClassificationHeader(charset_size=charset_size, input_embedding_size=input_embedding_size)
        self.max_word_length = max_word_length
        if decoder_type == "transformer":
            self.make_mask = torch.nn.Transformer().generate_square_subsequent_mask
            self.decoder_embedding = nn.Embedding(charset_size + 1, input_embedding_size)
            self.pos = PositionalEncoding(input_embedding_size, dropout_self_attention, max_len=max_word_length, positional_enconding_type=positional_enconding_type, init_pos_as_const=init_pos_as_const)
            self.transformer_decoder = nn.Sequential(*[nn.TransformerDecoderLayer(input_embedding_size, num_heads_self_attention, dropout=dropout_self_attention, batch_first=True)] * num_layers)

    def forward(self, embedding, target=None):
        if self.decoder_type == 'transformer':
            if target is not None:  # Teach-Forcing in training time
                targets_dec = target.clone()
                targets_dec[:, 1:] = targets_dec[:, :-1]
                targets_dec[:, 0] = self.charset_size
                tgt_emb = self.decoder_embedding(targets_dec)
                tgt_mask = self.make_mask(tgt_emb.size(1)).to(tgt_emb.device)
                for m in self.transformer_decoder:
                    tgt_emb = m(tgt_emb, embedding, tgt_mask)
                output = self.header(tgt_emb)
            else:
                output = torch.ones((embedding.shape[0], embedding.shape[1], self.charset_size)).long().to(embedding.device) * self.charset_size
                out_max = torch.ones((embedding.shape[0], embedding.shape[1])).long().to(embedding.device) * self.charset_size
                for t in range(1, embedding.shape[1]):
                    tgt_emb = self.pos(self.decoder_embedding(out_max[:, :t]))
                    tgt_mask = self.make_mask(t).to(embedding.device)
                    for m in self.transformer_decoder:
                        tgt_emb = m(tgt_emb, embedding, tgt_mask)
                    output_t = self.header(tgt_emb)
                    _, output_t_max = torch.max(output_t, dim=2)
                    out_max[:, t] = output_t_max[:, -1].long()
                    output[:, t, :] = output_t[:, -1, :]
        else:
            output = self.header(embedding)
        return output.permute((0, 2, 1))


class RecognitionModel(nn.Module):
    '''

    This model is not dependent in the input size!
    and Hence, we can decide the fixed input size or variable input size (and then we need to handle the batching)

    '''
    def __init__(self, charset_size, encoder_type='bilstm', decoder_type="transformer", hidden_size=256, num_layers=4, num_heads_self_attention=4,
                 dropout_self_attention=0.0, in_channels=3, max_word_length=26, positional_enconding_type='const', init_pos_as_const=True):
        super(RecognitionModel, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels)
        self.encoder = Encoder(encoder_type, feature_size=256, hidden_size=hidden_size, num_layers=num_layers,
                               num_heads_self_attention=num_heads_self_attention, dropout_self_attention=dropout_self_attention,
                               positional_enconding_type=positional_enconding_type, max_word_length=max_word_length, init_pos_as_const=init_pos_as_const)
        self.decoder = Decoder(charset_size + 1, input_embedding_size=hidden_size if encoder_type == 'transformer' else hidden_size * 2, decoder_type=decoder_type,
                               num_layers=num_layers, num_heads_self_attention=num_heads_self_attention, dropout_self_attention=dropout_self_attention,
                               positional_enconding_type=positional_enconding_type, max_word_length=max_word_length, init_pos_as_const=init_pos_as_const)

    def forward(self, word_image, target=None, return_embedding=False):
        features = self.feature_extractor(word_image)
        embedding = self.encoder(features)
        output = self.decoder(embedding, target)
        if return_embedding:
            return output, embedding
        return output

