from torch import nn

class Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(Classifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.layers = nn.Sequential(
            nn.Linear(embed_dim,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16, num_class)
        )

    def forward(self, text, offsets):
        embedding = self.embedding(text, offsets)
        x = self.layers(embedding)
        return x