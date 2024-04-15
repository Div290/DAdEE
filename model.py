import torch
import param
import torch.nn as nn
from transformers import BertModel, DistilBertModel, RobertaModel
from transformers.models.bert.modeling_bert import BertPooler
from transformers.models.roberta.modeling_roberta import RobertaPooler

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        feat = outputs[1]
        return feat


class DistilBertEncoder(nn.Module):
    def __init__(self):
        super(DistilBertEncoder, self).__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        feat = self.pooler(pooled_output)
        return feat


class RobertaEncoder(nn.Module):
    def __init__(self):
        super(RobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base')

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class DistilRobertaEncoder(nn.Module):
    def __init__(self):
        super(DistilRobertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('distilroberta-base')
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        feat = sequence_output[:, 0, :]
        return feat


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(param.hidden_size, param.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, x):
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaClassifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, dropout=0.1):
        super(RobertaClassifier, self).__init__()
        self.pooler = nn.Linear(param.hidden_size, param.hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(param.hidden_size, param.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.pooler(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        out = self.classifier(x)
        return out


    


class EarlyBertEncoder(nn.Module):
    def __init__(self):
        super(EarlyBertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.pooler = BertPooler(self.encoder.config)
        

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        hidden_states = outputs[2]  # Hidden states from all layers
        pooled_outputs = [self.pooler(layer_output) for layer_output in hidden_states]
        # print("The shape of pooled outputs is", len(pooled_outputs))
        
    # def __getitem__(self, index):
    #     return self.encoder[index]

        return pooled_outputs
    


class EarlyBertClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(EarlyBertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifiers = nn.ModuleList([nn.Linear(param.hidden_size, param.num_labels) for _ in range(param.num_exits)])
        self.apply(self.init_bert_weights)

    def forward(self, x):
        outputs = [classifier(self.dropout(x)) for classifier in self.classifiers]
        return outputs
    
    def __getitem__(self, index):
        return self.classifiers[index]

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.num_exits = param.num_exits
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(param.hidden_size, param.intermediate_size),
                nn.LeakyReLU(),
                nn.Linear(param.intermediate_size, param.intermediate_size),
                nn.LeakyReLU(),
                nn.Linear(param.intermediate_size, 1),
                nn.Sigmoid()
            ) for _ in range(param.num_exits)
        ])

    def forward(self, x):
        """Forward the discriminator."""
        outputs = [layer(x) for layer in self.layers]
        return outputs
    
    def __getitem__(self, index):
        return self.layers[index]
    
    
class EarlyRoBertaEncoder(nn.Module):
    def __init__(self):
        super(EarlyRoBertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained('roberta-base', output_hidden_states = True)
        self.pooler = RobertaPooler(self.encoder.config)
        

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        hidden_states = outputs[2]  # Hidden states from all layers
        # print("the shape", len(hidden_states))
        pooled_outputs = [self.pooler(layer_output) for layer_output in hidden_states]
        # print("The shape of pooled outputs is", len(pooled_outputs))
        
    # def __getitem__(self, index):
    #     return self.encoder[index]

        return pooled_outputs


class EarlyRoBertaClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(EarlyRoBertaClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifiers = nn.ModuleList([nn.Linear(param.hidden_size, param.num_labels) for _ in range(param.num_exits)])
        self.apply(self.init_bert_weights)

    def forward(self, x):
        outputs = [classifier(self.dropout(x)) for classifier in self.classifiers]
        return outputs
    
    def __getitem__(self, index):
        return self.classifiers[index]

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    