import torch
from torch import nn
from torch.nn import functional as F
import torch.nn as nn

from .beit import BEiT3Wrapper

class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim,
            norm_layer=norm_layer, 
        )
        self.pooler.apply(self._init_weights)
        #self.transition = nn.Linear(1024, 2048)###
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            norm_layer(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
        )
        self.head.apply(self._init_weights)

    def forward(self, input_tuple, output_answer_latent=False):
        image, question = input_tuple

        #combined_text = torch.cat([question, caption], dim=1)
        padding_mask = torch.zeros_like(question, dtype=torch.bool).cuda()
        outputs = self.beit3(
            textual_tokens=question, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
      
        x = outputs["encoder_out"]
        answer_latent = self.pooler(x)
        proj_feat = self.head(answer_latent)
        
        if output_answer_latent:
            return proj_feat, answer_latent

        return proj_feat


class BEiT3ForFinetune(BEiT3ForVisualQuestionAnswering):
    def __init__(self, __C, answer_size, base_answer_size=3129):
        super().__init__(__C, base_answer_size)
        #self.proj1 = nn.Linear(__C.FLAT_OUT_SIZE, answer_size - base_answer_size)
        self.proj1 = nn.Linear(1024, answer_size - base_answer_size)
        self._init_weights(self.proj1)
    # @torch.no_grad()
    # def parameter_init(self):
    #     self.proj1.weight.data.zero_()
    #     self.proj1.bias.data = self.proj1.bias.data.mean() + torch.zeros(self.proj1.bias.data.shape)
    def forward(self, input_tuple, output_answer_latent=False):
        proj_feat, answer_latent = super().forward(input_tuple, output_answer_latent=True)
        proj_feat = torch.cat([
            proj_feat,
            self.proj1(answer_latent)
        ], dim=1)
       
        if output_answer_latent:
            return proj_feat, answer_latent
        return proj_feat
