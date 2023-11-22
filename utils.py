import nltk
import torch
import pandas as pd
import matplotlib.pyplot as plt
from classifier import Classifier
from torchtext.vocab import build_vocab_from_iterator

def divide_df(df, porc_train = 0.8, porc_test = 0.2):
    count = df.count()[0]
    qtd_train = int(porc_train * count)
    qtd_test = int(porc_test * count)
    train_df = df.iloc[0:qtd_train]
    test_df = df.iloc[qtd_train:]
    reviews_list_train = list(train_df['review_text'])
    aval_list_train = list(train_df['overall_rating'])
    aval_list_train = fix_aval(aval_list_train)
    reviews_list_test = list(test_df['review_text'])
    aval_list_test = list(test_df['overall_rating'])
    aval_list_test = fix_aval(aval_list_test)
    return reviews_list_train, aval_list_train, reviews_list_test, aval_list_test

def fix_aval(list_aval):
    for i in range(len(list_aval)):
        a = list_aval[i]
        list_aval[i] = 1 if a in (1, 2) else 2 if a == 3 else 3
    return list_aval

def tokenize_reviews(data):
    newdata = []
    for label, text in data:
      newdata.append(nltk.word_tokenize(text.lower()))
    return newdata

def train(model, dataloader, optimizer, lossfunc):
    model.train()
    total_acc, total_loss = 0, 0
    for idx, (label, text, offsets) in enumerate(dataloader):
        pred = model(text, offsets)
        loss = lossfunc(pred, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_loss += loss.item()
        total_acc += (pred.argmax(1) == label).sum().item()
    total_acc = total_acc / len(dataloader.dataset)
    total_loss = total_loss / (len(dataloader))
    return total_acc, total_loss

def validate(model, dataloader, lossfunc):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            pred = model(text, offsets)
            loss = lossfunc(pred, label)
            total_loss += loss.item()
            total_acc += (pred.argmax(1) == label).sum().item()
    total_acc = total_acc / len(dataloader.dataset)
    total_loss = total_loss / (len(dataloader))
    return total_acc, total_loss 

def create_df(path, usecols):
    df = pd.read_csv(path, encoding='utf-8', usecols=usecols)
    # Remove caracteres indesejados
    df = df.replace(r'[,]|[.]|[!]|[(]|[)]|[?]', '', regex=True)
    # Remove reviews duplicados
    df = df.dropna().drop_duplicates()
    # Embaralha dataframe
    #df = df.sample(frac=1).reset_index(drop=True)
    return df

def plot(a, tam, b = None, title = None):
    enum = [i for i in range(tam)]
    plt.plot(enum, a)
    if b:
      plt.plot(enum, b)
    if title:
        plt.title(title)
    plt.show()

def load_model(path, vocab_size, device, emsize=128, num_class = 3):
    model = Classifier(vocab_size, emsize, num_class).to(device)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == "__main__":
    ...