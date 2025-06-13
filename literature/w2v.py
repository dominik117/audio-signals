from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

from nltk import tokenize

import nltk
nltk.download('punkt')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("alice_in_wonderland.txt")
data = fp.read()
sentences = [s.split() for s in tokenize.sent_tokenize(data)]

# train model (sg: 1 = skip gram, CBOW otherwise)
model = Word2Vec(sentences, min_count=1, sg=1)

# summarize the loaded model
print(model)

# summarize vocabulary
print(list(model.wv.key_to_index))

# access vector for one word
print(model.wv['Gryphon'])

# fit a 2d PCA model to the vectors
X = model.wv[model.wv.key_to_index]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])

words = list(model.wv.key_to_index)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

w1 = ['Queen']
model.wv.most_similar(positive=w1, topn=20)
