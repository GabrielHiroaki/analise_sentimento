from utils import *
import nltk
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch import nn

sentiment = ['Negativo', 'Neutro', 'Positivo']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nltk.download('punkt')
df = create_df('texto/B2W-Reviews01.csv', ["overall_rating", 'review_text'])
reviews_list_train, aval_list_train, reviews_list_test, aval_list_test = divide_df(df, 0.8, 0.2)
train_data = list(zip(aval_list_train,reviews_list_train))

# Vocabulario
print('Criando vocabulario\n')
train_data_token = tokenize_reviews(train_data)
vocab = build_vocab_from_iterator(train_data_token, specials=[""])
vocab.set_default_index(vocab[""])
print('Vocabulario criado\n')

# Funcao que transforma o tensor em uma frase
def tensor_to_text(tensor): # recebe um tensor como parametro
    words = []
    # percorre o tensor
    for i in tensor:
        w = vocab.lookup_token(i) # pega a palavra que corresponde ao seu id 
        words.append(w)
    return ' '.join(words)

# Funcao que classifica apenas uma frase
def classify_one_phrase(model, phrase):
    # Coloca a rede em modo de avaliacao
    model.eval()
    # Transforma a frase em um tensor
    text_tensor = torch.tensor(vocab(nltk.word_tokenize(phrase)), dtype=torch.int64)
    # Cria um offset
    # Como tenho apenas uma frase, nao preciso determinar o tamanho da mesma
    offsets = torch.tensor([0]).to(device)
    text_tensor = text_tensor.to(device)
    # Faco a transformacao inversa, de tensor para frase, dessa maneira, conseguimos ver
    # quais palavras estao em nosso vocabulario
    tt = tensor_to_text(text_tensor)
    print(f'Frase digitada: {tt}')
    with torch.no_grad():
        # Faz uma classificacao da nossa frase
        pred = model(text_tensor, offsets)
    # Retorna nossa predicao e o maior valor dentre elas (que e nossa classe)
    return pred, pred.argmax(1)

path = 'modelos/modelo1.pth'
model = load_model(path, len(vocab), device, emsize=128, num_class = 3)
# Organiza nossa predicao de modo que a soma de 1
softmax = nn.Softmax(dim=1)
print('Modelo carregado.\n')


while True:
    p = input('Digite sua frase: ').lower()
    a, b = classify_one_phrase(model, p)
    a = softmax(a)[0]
    print(f'Negativo: {a[0] * 100}%\n'
            f'Neutro: {a[1] * 100}%\n'
            f'Positivo: {a[2] * 100}%\n')
    print(f'Comentario {sentiment[b.item()]}')

    

