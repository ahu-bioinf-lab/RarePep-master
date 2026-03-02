import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = EncoderLayer(hid_dim = 128, kernel_size=3, dropout=0.2)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)  # (64,290,128)
        v_p = self.protein_extractor(v_p)  # (64,1200)-->(64, 1185, 128)
        f, att = self.bcn(v_d, v_p)  # f:(32,256), att:(32,2,290,1185)
        score = self.mlp_classifier(f)  # (32,2)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        #talking heads
        self.pre_softmax_talking_heads = nn.Conv2d(n_heads, n_heads, 1, bias=False)
        self.post_softmax_talking_heads = nn.Conv2d(n_heads, n_heads, 1, bias=False)
        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        energy = self.pre_softmax_talking_heads(energy)
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        attention = self.post_softmax_talking_heads(attention)
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.conv = nn.Conv1d(32, 256, kernel_size, padding=(kernel_size - 1) // 2)  # for _ in range(self.n_layers)])  # convolutional layers
        self.dropout_layer = nn.Dropout(dropout)
        self.ft = nn.Linear(32, 128)
        self.sa = SelfAttention(hid_dim, 8, dropout, device)
        self.do = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, conv_input):
        conv_input = conv_input.to(device)
        batch = conv_input.shape[0]  # 保留批量大小，即64
        conv_input = conv_input.reshape(batch, -1, 32)  # 根据 hid_dim 来重塑形状
        global_input = self.ft(conv_input)  # (b,24,128)
        global_output = self.ln(global_input + self.do(self.sa(global_input, global_input, global_input)))
        # 执行卷积操作前，需要转置输入张量的最后两个维度，因为 Conv1d 的要求
        # conv_input = conv_input.permute(0, 2, 1)  # 转置后形状为 (batch, hid_dim, seq_len)，这符合 Conv1d 的输入要求
        # conved = self.conv(self.dropout_layer(conv_input))  # 执行卷积操作
        # conved = F.glu(conved, dim=1)  # 使用 GLU 非线性变换，将特征通道数减半
        # # conved = conved + conv_input  # 残差连接
        # conved_output = conved.permute(0, 2, 1)  # 转置回 (batch, seq_len, hid_dim)，方便后续处理
        # output = global_output + conved_output

        return global_output

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size

        self.conv11 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.bn11 = nn.BatchNorm1d(in_ch[1])
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding=1)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding=4)
        self.bn3 = nn.BatchNorm1d(in_ch[3])
        self.fc = nn.Linear(2, 128)
        self.conv31 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.bn31 = nn.BatchNorm1d(128)
        self.conv61 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.bn61 = nn.BatchNorm1d(128)
        self.conv91 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1, padding=0)
        self.bn91 = nn.BatchNorm1d(128)
        self.se_block = SEBlock(128)
        self.softmax = nn.Softmax(dim=1)

    # def forward(self, v):
    #     b = v.shape[0]
    #     v = v.reshape(b, -1, 2)  # (64,640,2)
    #     v = self.fc(v)  # (64,640,128)
    #     # v = self.embedding(v.long())  # (64,1280)
    #     v = v.transpose(2, 1)  # (64,128,640)
    #     v_resdu = v
    #     v3 = self.bn1(F.relu(self.conv1(v)))
    #     v_c3 = self.se_block(v3)
    #     attention_vectors = self.softmax(v_c3)  # (64,384,640)
    #     v_cc3 = v3 * attention_vectors  #(64,384,640)
    #     v6 = self.bn2(F.relu(self.conv2(v_cc3)))
    #     v_c6 = self.se_block(v6)
    #     attention_vectors = self.softmax(v_c6)  # (64,384,640)
    #     v_cc6 = v6 * attention_vectors  # (64,384,640)
    #     v9 = self.bn3(F.relu(self.conv3(v_cc6)))
    #     v_c9 = self.se_block(v9)
    #     attention_vectors = self.softmax(v_c9)  # (64,384,640)
    #     v_cc9 = v9 * attention_vectors  # (64,384,640)
    #     # # 1×1
    #     # v = self.bn11(F.relu(self.conv11(v)))  # (64,128,640)
    #     # # 3×3
    #     # v3 = self.bn1(F.relu(self.conv1(v)))  # (64,128,640)
    #     # # 6×6
    #     # v6 = self.bn2(F.relu(self.conv2(v)))  # (64,128,640)
    #     # # 9×9
    #     # v9 = self.bn3(F.relu(self.conv3(v)))  # (64,128,640)
    #     #
    #     # v31 = self.bn31(F.relu(self.conv31(v3)))
    #     # v61 = self.bn61(F.relu(self.conv61(v6)))
    #     # v91 = self.bn91(F.relu(self.conv91(v9)))
    #     v_f = v_cc9+v_resdu  # v31+v61+v91
    #     v_f = v_f.view(v_f.size(0), v_f.size(2), -1)
    #     return v_f
    def forward(self, v):
        b = v.shape[0]
        v = v.reshape(b, -1, 2)  # (64,640,2)
        v = self.fc(v)  # (64,640,128)
        # v = self.embedding(v.long())  # (64,1280)
        v = v.transpose(2, 1)  # (64,128,640)
        v_resdu = v
        # 1×1
        v = self.bn11(F.relu(self.conv11(v)))  # (64,128,640)
        # 3×3
        v3 = self.bn1(F.relu(self.conv1(v)))  # (64,128,640)
        # 6×6
        v6 = self.bn2(F.relu(self.conv2(v)))  # (64,128,640)
        # 9×9
        v9 = self.bn3(F.relu(self.conv3(v)))  # (64,128,640)

        # v_c = torch.cat([v3, v6, v9], dim=1)
        # v_c3 = self.se_block(v3)
        # v_c6 = self.se_block(v6)
        # v_c9 = self.se_block(v9)

        # v_se = torch.cat([v_c3, v_c6, v_c9], dim=1)  # (64,384,640)
        # attention_vectors = self.softmax(v_se)  # (64,384,640)
        # v_f = v_c * attention_vectors  #(64,384,640)

        v31 = self.bn31(F.relu(self.conv31(v3)))
        v31 = v31 + v_resdu
        v61 = self.bn61(F.relu(self.conv61(v6)))
        v61 = v61 + v_resdu
        v91 = self.bn91(F.relu(self.conv91(v9)))
        v91 = v91 + v_resdu
        v_f = v31+v61+v91
        v_fe = self.se_block(v_f)
        attention_vectors = self.softmax(v_fe)  # (64,384,640)
        v_f = v_f * attention_vectors  #(64,384,640)
        v_f = v_f.view(v_f.size(0), v_f.size(2), -1)  # (64,384,640)
        return v_f


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
