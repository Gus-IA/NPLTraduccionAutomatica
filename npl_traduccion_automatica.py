import unicodedata
import re
import random
import torch
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_file(file, reverse=False):
    # Read the file and split into lines
    lines = open(file, encoding="utf-8").read().strip().split("\n")

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")[:2]] for l in lines]

    return pairs


pairs = read_file("spa.txt")

print(random.choice(pairs))

SOS_token = 0
EOS_token = 1
PAD_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "PAD": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS, EOS and PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(" ")]

    def sentenceFromIndex(self, index):
        return [self.index2word[ix] for ix in index]


MAX_LENGTH = 10

eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)


def filterPair(p, lang, filters, max_length):
    return (
        len(p[0].split(" ")) < max_length
        and len(p[1].split(" ")) < max_length
        and p[lang].startswith(filters)
    )


def filterPairs(pairs, filters, max_length, lang=0):
    return [pair for pair in pairs if filterPair(pair, lang, filters, max_length)]


def prepareData(file, filters=None, max_length=None, reverse=False):

    pairs = read_file(file, reverse)
    print(f"Tenemos {len(pairs)} pares de frases")

    if filters is not None:
        assert max_length is not None
        pairs = filterPairs(pairs, filters, max_length, int(reverse))
        print(f"Filtramos a {len(pairs)} pares de frases")

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang("eng")
        output_lang = Lang("spa")
    else:
        input_lang = Lang("spa")
        output_lang = Lang("eng")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

        # add <eos> token
        pair[0] += " EOS"
        pair[1] += " EOS"

    print("Longitud vocabularios:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData("spa.txt")

# descomentar para usar el dataset filtrado
# input_lang, output_lang, pairs = prepareData('spa.txt', filters=eng_prefixes, max_length=MAX_LENGTH)

print(random.choice(pairs))

print(output_lang.indexesFromSentence("tengo mucha sed ."))

print(output_lang.sentenceFromIndex([68, 5028, 135, 4]))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ix):
        return torch.tensor(
            self.input_lang.indexesFromSentence(self.pairs[ix][0]),
            device=device,
            dtype=torch.long,
        ), torch.tensor(
            self.output_lang.indexesFromSentence(self.pairs[ix][1]),
            device=device,
            dtype=torch.long,
        )

    def collate(self, batch):
        # calcular longitud máxima en el batch
        max_input_len, max_output_len = 0, 0
        for input_sentence, output_sentence in batch:
            max_input_len = (
                len(input_sentence)
                if len(input_sentence) > max_input_len
                else max_input_len
            )
            max_output_len = (
                len(output_sentence)
                if len(output_sentence) > max_output_len
                else max_output_len
            )
        # añadimos padding a las frases cortas para que todas tengan la misma longitud
        input_sentences, output_sentences = [], []
        for input_sentence, output_sentence in batch:
            input_sentences.append(
                torch.nn.functional.pad(
                    input_sentence,
                    (0, max_input_len - len(input_sentence)),
                    "constant",
                    self.input_lang.word2index["PAD"],
                )
            )
            output_sentences.append(
                torch.nn.functional.pad(
                    output_sentence,
                    (0, max_output_len - len(output_sentence)),
                    "constant",
                    self.output_lang.word2index["PAD"],
                )
            )
        # opcionalmente, podríamos re-ordenar las frases en el batch (algunos modelos lo requieren)
        return torch.stack(input_sentences), torch.stack(output_sentences)


# separamos datos en train-test
train_size = len(pairs) * 80 // 100
train = pairs[:train_size]
test = pairs[train_size:]

dataset = {
    "train": Dataset(input_lang, output_lang, train),
    "test": Dataset(input_lang, output_lang, test),
}

print(len(dataset["train"]), len(dataset["test"]))

input_sentence, output_sentence = dataset["train"][1]

print(input_sentence, output_sentence)

print(
    input_lang.sentenceFromIndex(input_sentence.tolist()),
    output_lang.sentenceFromIndex(output_sentence.tolist()),
)

dataloader = {
    "train": torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=32,
        shuffle=True,
        collate_fn=dataset["train"].collate,
    ),
    "test": torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=256,
        shuffle=False,
        collate_fn=dataset["test"].collate,
    ),
}

inputs, outputs = next(iter(dataloader["train"]))
print(inputs.shape, outputs.shape)


class Encoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size=100, hidden_size=100, n_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.gru = torch.nn.GRU(
            embedding_size, hidden_size, num_layers=n_layers, batch_first=True
        )

    def forward(self, input_sentences):
        embedded = self.embedding(input_sentences)
        output, hidden = self.gru(embedded)
        # del encoder nos interesa el último *hidden state*
        return hidden


encoder = Encoder(input_size=input_lang.n_words)
hidden = encoder(torch.randint(0, input_lang.n_words, (64, 10)))

# [num layers, batch size, hidden size]
print(hidden.shape)


class Decoder(torch.nn.Module):
    def __init__(self, input_size, embedding_size=100, hidden_size=100, n_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_size, embedding_size)
        self.gru = torch.nn.GRU(
            embedding_size, hidden_size, num_layers=n_layers, batch_first=True
        )
        self.out = torch.nn.Linear(hidden_size, input_size)

    def forward(self, input_words, hidden):
        embedded = self.embedding(input_words)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden


decoder = Decoder(input_size=output_lang.n_words)
output, decoder_hidden = decoder(torch.randint(0, output_lang.n_words, (64, 1)), hidden)

# [batch size, vocab size]
print(output.shape)

# [num layers, batch size, hidden size]
print(decoder_hidden.shape)


def fit(encoder, decoder, dataloader, epochs=10):
    encoder.to(device)
    decoder.to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        train_loss = []
        bar = tqdm(dataloader["train"])
        for batch in bar:
            input_sentences, output_sentences = batch
            bs = input_sentences.shape[0]
            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # obtenemos el último estado oculto del encoder
            hidden = encoder(input_sentences)
            # calculamos las salidas del decoder de manera recurrente
            decoder_input = torch.tensor(
                [[output_lang.word2index["SOS"]] for b in range(bs)], device=device
            )
            for i in range(output_sentences.shape[1]):
                output, hidden = decoder(decoder_input, hidden)
                loss += criterion(output, output_sentences[:, i].view(bs))
                # el siguiente input será la palbra predicha
                decoder_input = torch.argmax(output, axis=1).view(bs, 1)
            # optimización
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            train_loss.append(loss.item())
            bar.set_description(
                f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f}"
            )

        val_loss = []
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            bar = tqdm(dataloader["test"])
            for batch in bar:
                input_sentences, output_sentences = batch
                bs = input_sentences.shape[0]
                loss = 0
                # obtenemos el último estado oculto del encoder
                hidden = encoder(input_sentences)
                # calculamos las salidas del decoder de manera recurrente
                decoder_input = torch.tensor(
                    [[output_lang.word2index["SOS"]] for b in range(bs)], device=device
                )
                for i in range(output_sentences.shape[1]):
                    output, hidden = decoder(decoder_input, hidden)
                    loss += criterion(output, output_sentences[:, i].view(bs))
                    # el siguiente input será la palbra predicha
                    decoder_input = torch.argmax(output, axis=1).view(bs, 1)
                val_loss.append(loss.item())
                bar.set_description(
                    f"Epoch {epoch}/{epochs} val_loss {np.mean(val_loss):.5f}"
                )


fit(encoder, decoder, dataloader, epochs=30)

input_sentence, output_sentence = dataset["train"][129]
input_lang.sentenceFromIndex(input_sentence.tolist()), output_lang.sentenceFromIndex(
    output_sentence.tolist()
)

(["come", "in", ".", "EOS"], ["pase", ".", "EOS"])


def predict(input_sentence):
    # obtenemos el último estado oculto del encoder
    hidden = encoder(input_sentence.unsqueeze(0))
    # calculamos las salidas del decoder de manera recurrente
    decoder_input = torch.tensor([[output_lang.word2index["SOS"]]], device=device)
    # iteramos hasta que el decoder nos de el token <eos>
    outputs = []
    while True:
        output, hidden = decoder(decoder_input, hidden)
        decoder_input = torch.argmax(output, axis=1).view(1, 1)
        outputs.append(decoder_input.cpu().item())
        if decoder_input.item() == output_lang.word2index["EOS"]:
            break
    return output_lang.sentenceFromIndex(outputs)


print(predict(input_sentence))
