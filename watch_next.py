mport spacy

nlp = spacy.load("en_core_web_md")

tokens = nlp(u'Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator.')

for token in tokens:

print(token.text, token.has_vector, token.vector_norm, token.is_oov)

labels = [a.text for a in tokens]

print(labels)

M = np.zeros((len(tokens), len(tokens)))

for idx, token1 in enumerate(tokens):

for idy, token2 in enumerate(tokens):

M[idx, idy] = token1.similarity(token2)

%matplotlib inline

import numpy as np

import seaborn as sns

import matplotlib.pylab as plt

ax = sns.heatmap(M, cmap = "RdBu_r", xticklabels=labels, yticklabels=labels)

plt.show()