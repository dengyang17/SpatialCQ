from transformers import BertModel, RobertaModel
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

model = {'bert': BertModel, 'roberta': RobertaModel}
hidden_size = {'bert': 768, 'roberta': 1024}

class SpatialRank(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.encoder = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
        self.dropout = nn.Dropout(0.5)
        args.hidden_size = config.hidden_size
        self.rgat = RGAT(args)
        self.classifier = nn.Linear(args.hidden_size, 2)
        self.config = config

    def forward(self, input_ids, attention_mask, state_nodes, state_adjs, labels=None):
        node_embed = self.encoder.embeddings(state_nodes)
        node_embed = self.rgat(node_embed, state_adjs)
        node_mask = state_nodes.ne(0)

        inputs_embeds = self.encoder.embeddings(input_ids)
        inputs_embeds = torch.cat((inputs_embeds, node_embed), dim=1)

        attention_mask = torch.cat((attention_mask, node_mask), dim=1)

        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)


class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, att_dim, hidden_size):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.att_dim = att_dim
        self.leakyrelu = nn.LeakyReLU(1e-2)

        self.W = nn.Linear(in_features, att_dim)

        a_layers = [nn.Linear(2 * att_dim, hidden_size), 
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

    def forward(self, input, adj):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)

        h = self.W(input) # (B, N, D)
        a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
        e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
        attention = F.softmax(mask_logits(e, dmask), dim=1)
        attention = attention.view(*adj.size())

        output = attention.bmm(h)
        return output

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)



class RGAT(nn.Module):
    def __init__(self, config):# nfeat, nhid, nclass, nlayer, nrel, att_dim, hidden_size, dropout
        super(RGAT, self).__init__()
        self.num_hops = config.num_hops
        self.rgats = nn.ModuleList()
        for j in config.relations:
            gats = nn.ModuleList()
            for i in range(config.num_hops):
                gats.append(GATLayer(config.hidden_size, config.hidden_size, config.hidden_size, config.hidden_size))
            self.rgats.append(gats)
        self.dropout = config.dropout
        self.wo = nn.Linear(config.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adjs):
        adjs = adjs.permute(1,0,2,3)
        for i in range(self.num_hops-1):
            x = torch.stack([self.rgats[j][i](x, adjs[j]) for j in range(len(self.rgats))], dim=2)
            x = F.relu(torch.sum(x, dim=2))
            x = F.dropout(x, self.dropout, training=self.training)
        #return F.relu(torch.sum(torch.stack([self.rgats[j][-1](x, adjs[j]) for j in range(len(self.rgats))], dim=2), dim=2))

        x = F.relu(torch.sum(torch.stack([self.rgats[j][-1](x, adjs[j]) for j in range(len(self.rgats))], dim=2), dim=2))

        return x