from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
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
    with torch.no_grad():
        # Faz uma classificacao da nossa frase
        pred = model(text_tensor, offsets)
    # Retorna nossa predicao e o maior valor dentre elas (que e nossa classe)
    return pred, pred.argmax(1)

# Carregando modelo treinado anteriormente
path = 'modelos/modelo1.pth'
model = load_model(path, len(vocab), device, emsize=128, num_class = 3)
# Organiza nossa predicao de modo que a soma de 1
softmax = nn.Softmax(dim=1)
print('Modelo carregado.\n')

show_perc = False

def echo(update, context):
    a, b = classify_one_phrase(model, update.message.text.lower())
    a = softmax(a)[0]
    if show_perc:
        message = f'Negativo: {a[0] * 100:.2f}%\n \
Neutro: {a[1] * 100:.2f}%\n \
Positivo: {a[2] * 100:.2f}%\n'
    else:
        message = f'Comentario {sentiment[b.item()]}'
    update.message.reply_text(message)

# Resposta para o comando /help
def help(update, context):
    update.message.reply_text('Classifico uma mensagem como positiva ou negativa, basta digitar qualquer coisa. Para alternar meu modo de saida, digite /change_mode')

# Resposta para erros
def error(update, context):
    update.message.reply_text('Ocorreu um erro.')

def change_mode(update, context):
    global show_perc
    update.message.reply_text('Ok, alterando meu tipo de saida.')
    show_perc = not show_perc

token = "5426913176:AAFBveOEwxYe3XIKCfJS9pbDnAvcQjgb3Jg"
updater = Updater(token, use_context=True)
dp = updater.dispatcher
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("change_mode", change_mode))
dp.add_handler(MessageHandler(Filters.text, echo))
dp.add_error_handler(error)

print('Bot ativo...')
updater.start_polling()
updater.idle()
