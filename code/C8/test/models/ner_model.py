import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from .base import BaseNerNetwork

class BiGRUNerNetWork(BaseNerNetwork):
    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=1):
        super(BiGRUNerNetWork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.gru_layers = nn.ModuleList()
        for _ in range(num_gru_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )
            )
        
        self.fc = nn.Linear(2 * hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu()
        embedded_text = self.embedding(token_ids)

        current_input = embedded_text
        for gru_layer in self.gru_layers:
            packed_input = rnn.pack_padded_sequence(
                current_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = gru_layer(packed_input)
            output, _ = rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=token_ids.shape[1]
            )
            
            output = self.fc(output)
            current_input = current_input + output

        logits = self.classifier(current_input)
        return logits
