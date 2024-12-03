import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils
import pickle
from datasets import load_dataset
import config
from collections import defaultdict



def print_system_info():
    virtual_memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {virtual_memory.percent}% ({virtual_memory.used // (1024**2)} MB / {virtual_memory.total // (1024**2)} MB)")

    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() // (1024**2)
        gpu_memory_reserved = torch.cuda.memory_reserved() // (1024**2)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"GPU Memory: {gpu_memory_allocated} MB allocated, {gpu_memory_reserved} MB reserved, {gpu_memory_total} MB total")
    else:
        print("No GPU available.")


# Load data from Huggingface
dataset = load_dataset(config.DATASET_NAME, delimiter="\t")
# Split data into training, validation, and test sets using parameter from config.py
data = [(item["unnormalized_string"], item["normalized_string"]) for item in dataset["train"]]
train_data, test_data = train_test_split(data, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
val_data, test_data = train_test_split(test_data, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED)

# Build vocabulary
vocab = defaultdict(int)
for unnormalized, normalized in data:
    for char in unnormalized + normalized:
        vocab[char] += 1

# Add special tokens
vocab = {char: idx for idx, char in enumerate(vocab.keys(), start=len(config.SPECIAL_TOKENS))}
vocab.update(config.SPECIAL_TOKENS)


def encode(word, vocab, add_sos_eos=False):
    tokens = [vocab.get(char, vocab["<unk>"]) for char in word]
    if add_sos_eos:
        tokens = [vocab["<sos>"]] + tokens + [vocab["<eos>"]]
    return tokens


def decode(indices, vocab):
    inv_vocab = {idx: char for char, idx in vocab.items()}
    chars = []
    for idx in indices:
        if idx == vocab["<eos>"]:
            break
        if idx != vocab["<pad>"] and idx != vocab["<sos>"]:
            chars.append(inv_vocab.get(idx, ""))
    return "".join(chars)


class ChunkedDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unnormalized, normalized = self.data[idx]
        src = torch.tensor(encode(unnormalized, self.vocab), dtype=torch.long)
        trg = torch.tensor(encode(normalized, self.vocab, add_sos_eos=True), dtype=torch.long)
        return src, trg


train_dataset = ChunkedDataset(train_data, vocab)
val_dataset = ChunkedDataset(val_data, vocab)
test_dataset = ChunkedDataset(test_data, vocab)

padding_idx = vocab["<pad>"]


def pad_batch(batch):
    src, trg = zip(*batch)
    src_padded = pad_sequence(src, batch_first=True, padding_value=padding_idx)
    trg_padded = pad_sequence(trg, batch_first=True, padding_value=padding_idx)
    return src_padded, trg_padded


train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=pad_batch,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=pad_batch,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=pad_batch,
)


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

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5, max_length=100):
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

        if trg is not None:
            # Training mode
            seq_len = trg.size(1)
            outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(src.device)
            input_step = trg[:, 0].unsqueeze(1)  # Start token

            for t in range(1, seq_len):
                attn_weights = self.attention(decoder_hidden[-1], enc_output)
                context = torch.bmm(attn_weights.unsqueeze(1), enc_output).squeeze(1)
                input_step_embedded = self.embedding(input_step)
                projected_input = self.decoder_input_projection(
                    torch.cat((input_step_embedded.squeeze(1), context), dim=1)
                )
                dec_input = projected_input.unsqueeze(1)
                dec_output, (decoder_hidden, decoder_cell) = self.decoder(dec_input, (decoder_hidden, decoder_cell))
                outputs[:, t, :] = self.fc(dec_output.squeeze(1))
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = outputs[:, t, :].argmax(1).unsqueeze(1)
                input_step = trg[:, t].unsqueeze(1) if teacher_force and t < seq_len else top1
        else:
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


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=0.5)
            # Align output to target sequence length
            output = output[:, :trg.size(1), :]  # Ensure output matches target length
            # Flatten for CrossEntropyLoss
            output_flat = output.view(-1, output.shape[-1])  # Shape: [batch_size * seq_len, vocab_size]
            trg_flat = trg.view(-1)  # Shape: [batch_size * seq_len]
            # Compute loss
            loss = criterion(output_flat, trg_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

def evaluate_model(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            output = output[:, :trg.size(1), :]
            loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_system_info()

input_dim = output_dim = len(vocab)
model = Seq2SeqWithAttention(
    input_dim,
    config.EMBEDDING_DIM,
    config.HIDDEN_DIM,
    output_dim,
    padding_idx,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.SCHEDULER_MODE, patience=config.SCHEDULER_PATIENCE, verbose=config.SCHEDULER_VERBOSE)

train_model(model, train_loader, val_loader, optimizer, criterion, scheduler)

test_loss = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")

# Saving the model
torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

# Save the vocabulary
with open(config.VOCAB_SAVE_PATH, "wb") as f:
    pickle.dump(vocab, f)
