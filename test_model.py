import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import pickle
import json
from huggingface_hub import hf_hub_download

# Define the same model architecture
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention, dim=1)

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, padding_idx, num_layers=3, dropout=0.3):
        super(Seq2SeqWithAttention, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention(hidden_dim)
        self.decoder_input_projection = nn.Linear(emb_dim + hidden_dim * 2, hidden_dim * 2)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.0, max_length=100):
        src_lengths = (src != self.embedding.padding_idx).sum(dim=1)
        embedded = self.dropout(self.embedding(src))
        packed_embedded = rnn_utils.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.encoder(packed_embedded)
        enc_output, _ = rnn_utils.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=self.embedding.padding_idx
        )

        batch_size = src.size(0)
        hidden_dim = hidden.size(2)

        # Combine bidirectional encoder hidden states
        hidden = hidden.view(self.encoder.num_layers, 2, batch_size, hidden_dim)
        cell = cell.view(self.encoder.num_layers, 2, batch_size, hidden_dim)
        decoder_hidden = hidden.sum(dim=1)
        decoder_cell = cell.sum(dim=1)

        outputs = []

        # Inference mode
        input_step = torch.tensor([vocab['<sos>']] * batch_size, dtype=torch.long, device=src.device).unsqueeze(1)
        for _ in range(max_length):
            attn_weights = self.attention(decoder_hidden[-1], enc_output)
            context = torch.bmm(attn_weights.unsqueeze(1), enc_output).squeeze(1)
            input_step_embedded = self.embedding(input_step)
            projected_input = self.decoder_input_projection(
                torch.cat((input_step_embedded.squeeze(1), context), dim=1)
            )
            dec_input = projected_input.unsqueeze(1)
            dec_output, (decoder_hidden, decoder_cell) = self.decoder(dec_input, (decoder_hidden, decoder_cell))
            output_token = self.fc(dec_output.squeeze(1))
            outputs.append(output_token.unsqueeze(1))
            top1 = output_token.argmax(1).unsqueeze(1)
            input_step = top1
            if (top1 == vocab['<eos>']).all():
                break
        outputs = torch.cat(outputs, dim=1)

        return outputs

def encode(word, vocab):
    return [vocab.get(char, vocab['<unk>']) for char in word]

def decode(indices, vocab):
    inv_vocab = {idx: char for char, idx in vocab.items()}
    chars = []
    for idx in indices:
        if idx == vocab['<eos>']:
            break
        if idx != vocab['<pad>'] and idx != vocab['<sos>']:
            chars.append(inv_vocab.get(idx, ''))
    return ''.join(chars)

def test_model(model, input_str, vocab):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([encode(input_str, vocab)], dtype=torch.long).to(device)
        output = model(input_tensor, trg=None, teacher_forcing_ratio=0.0)
        output_indices = output.argmax(dim=-1).squeeze().cpu().numpy()
        return decode(output_indices, vocab)

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model, vocabulary, and config from Hugging Face
    repo_id = "mschonhardt/georges-1913-normalization-model"
    model_path = hf_hub_download(repo_id=repo_id, filename="normalization_model.pth")
    vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.pkl")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    # Load the vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load the config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    padding_idx = vocab['<pad>']
    input_dim = output_dim = len(vocab)
    emb_dim = config["model_parameters"]["embedding_dim"]
    hidden_dim = config["model_parameters"]["hidden_dim"]
    num_layers = config["model_parameters"]["num_layers"]
    dropout = config["model_parameters"]["dropout"]
    max_length = config["model_parameters"]["max_length"]

    # Initialize model
    model = Seq2SeqWithAttention(
        input_dim,
        emb_dim,
        hidden_dim,
        output_dim,
        padding_idx,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test the model with text
    with open("./test_normalisation.txt", "r", encoding="utf8") as test_file:
        test_strs = test_file.read()
    test_strs = test_strs.split()

    normalized_words = []

    for test_str in test_strs[:100]:
        output_str = test_model(model, test_str.lower(), vocab)
        print(f"Input: {test_str}, Output: {output_str}")
        normalized_words.append(output_str)
