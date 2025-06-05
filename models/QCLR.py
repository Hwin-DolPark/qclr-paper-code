import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, final=False):
        super().__init__()
        # two dilated Conv1d + GELU, padding='same'
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding='same', dilation=dilation),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding='same', dilation=dilation),
            nn.GELU(),
        )
        # 1×1 projector on channel mismatch or last block
        self.projector = (
            nn.Conv1d(in_ch, out_ch, 1)
            if (in_ch != out_ch or final) else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.projector(x)


class DilatedConv(nn.Module):
    def __init__(self, in_ch, channels, kernel_size):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_prev = in_ch if i == 0 else channels[i-1]
            layers.append(
                ConvBlock(
                    in_ch=in_prev,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels)-1)
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # projection head for finetune
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        if self.output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x


class Model(nn.Module):
    """
    QCLR Paper link: https://
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding - configs.d_model = 128
        self.input_fc = nn.Linear(configs.enc_in, int(configs.d_model/2))
        self.depth = 5
        self.dil_conv = DilatedConv(
            int(configs.d_model/2),
            [int(configs.d_model/2)] * self.depth + [configs.d_model],
            kernel_size=3
        )
        self.sim_output_dim = configs.sim_dim #96

        # Similarity Projector Module - Representation Vector
        self.proj_head_representation = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.BatchNorm1d(configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, self.sim_output_dim)
        )

        # Mortality Decoder Module - measure clinical outcome
        self.proj_head_clinical_outcome = ProjectionHead(configs.d_model, configs.num_class,
                                        configs.d_model)

    def medical_encoder(self, x_enc, x_mark_enc):
        x = self.input_fc(x_enc)
        x = x.transpose(1, 2)
        x = self.dil_conv(x)
        x = x.transpose(1, 2)

        # z - Encoded Representation
        encoded_representation = F.max_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).transpose(1, 2)
        out = encoded_representation.squeeze(1)

        # v - Representation Vector
        representation_vector = self.proj_head_representation(out)
        # y^hat - Predicted Mortality
        output = self.proj_head_clinical_outcome(out)

        return (output, representation_vector)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "classification_qclr" or self.task_name == "classification_qclr_asan":
            dec_out = self.medical_encoder(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == "classification_asan":
            dec_out = self.medical_encoder(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
