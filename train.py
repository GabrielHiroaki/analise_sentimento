from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch
import nltk
from torch import nn
from classifier import Classifier
from utils import *
from collections import Counter


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}\n')
nltk.download('punkt')

# Obtenho as colunas "overall_rating" e "review_text" do arquivo csv utilizando a biblioteca pandas
df = create_df('texto/B2W-Reviews01.csv', ['overall_rating', 'review_text'])
# Divido os dados entre o conjunto de treino e validacao
reviews_list_train, aval_list_train, reviews_list_test, aval_list_test = divide_df(df, 0.8, 0.2)
print(f'\nTamanho do conjunto de treino: {len(reviews_list_train)} reviews')
print(f'Tamanho do conjunto de validacao: {len(reviews_list_test)} reviews')

count = Counter(aval_list_train)
print(f'\nQuantidade por classificacao: \n'
        f'{dict(count)}')

# Crio uma unica lista contendo as avaliacoes (nota que vai de 1 a 5) e os reviews (textos)
# Treino
train_data = list(zip(aval_list_train,reviews_list_train))
# Validacao
valid_data = list(zip(aval_list_test,reviews_list_test))

# Construindo o vocabulario
# Separo minhas frases em tokens
train_data_token = tokenize_reviews(train_data)
# Crio o vocabulario
"""
    Segue o principio de um dicionario, onde minhas chaves serao
    as palavras que fazem parte do meu dicionario e os valores
    serao os indices dessas chaves. Ex:
    {"palavra1": 1, "palavra2": 2, ...}
""" 
vocab = build_vocab_from_iterator(train_data_token, specials=[""])
# A chave do indice padrao e uma string vazia, seu valor = 0
# Sendo assim, vocab[""] = 0
vocab.set_default_index(vocab[""])

# Funcao responsavel por transformar meu batch em tensor, ajustando tambem seu tamanho para passar pelo modelo
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for label, text in batch:
        # Possuo 3 classes: 1, 2, 3 porem, o modelo entende a partir do zero
        # sendo assim, minhas classes serao 0, 1, 2
        label_list.append(int(label)-1)
        text_tensor = torch.tensor(vocab(nltk.word_tokenize(text)), dtype=torch.int64)
        text_list.append(text_tensor)
        # pega o tamanho do review (quantidade de palavras do review)
        offsets.append(text_tensor.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # Retorna a soma acumulada dos elementos do offsetts (excluindo o ultimo elemento) na dimensao 0
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # Concatena os tensores do batch
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# parametros do modelo
num_class = 3
vocab_size = len(vocab)
print(f'Tamanho do vocabulario: {vocab_size} palavras.')
emsize = 128
# cria o modelo
model = Classifier(vocab_size, emsize, num_class).to(device)

# Hiperparametros
epochs = 20
# Taxa de aprendizagem
lr = 1
# Tamanho do batch
batch_size = 256
# Funcao de perda
lossfunc = nn.CrossEntropyLoss()
# Otimizador
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Criando os dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)

print(f'Iniciando o treinamento')
accs_train = []
losses_train = []
accs_aval = []
losses_aval = []

best_acc = 0
tol = 0.005
c = 0
patience = 3
path = 'modelos/'
name_model = 'modelo1'

for epoch in range(1, epochs + 1):
    acc_train, loss_train = train(model, train_dataloader, optimizer, lossfunc)
    acc_aval, loss_aval = validate(model, valid_dataloader, lossfunc)
    accs_train.append(acc_train)
    losses_train.append(loss_train)
    accs_aval.append(acc_aval)
    losses_aval.append(loss_aval)
    print(f'Epoca: {epoch}\n'
            f'Perda no treino: {loss_train}, Acuracia no treino: {acc_train}\n'
            f'Perda na validacao: {loss_aval}, Acuracia na validacao: {acc_aval}\n')
    if acc_aval > best_acc + tol:
        # Salva o modelo na pasta modelos
        torch.save(model.state_dict(), path + name_model + '.pth')
        print(f'Melhor modelo salvo em {path}')
        best_acc = acc_aval
        c = 0
    else:
        c += 1
    if c == patience:
        print(f'O modelo teve seu treinamento interrompido pois nao houve melhoras ha {c} epocas.')
        break

# Mostra algumas metricas utilizando matplotlib
plot(losses_train, len(losses_train), losses_aval, "Perdas")
plot(accs_train, len(accs_train), accs_aval, "Acuracias")