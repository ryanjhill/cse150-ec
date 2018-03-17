import markovify
import random
import numpy as np

state_size = 6
harry_potter = "./Harry_Potter_Sorcerers_Stone.txt"
test_samples = 10000

text = markovify.Text(open(harry_potter).read(), state_size = state_size)

def sample(text, state_size=2):
    words = ' '.join([i.strip() for i in open(text).readlines() if i]).split()
    while True:
        index = random.randint(0, len(words))
        yield tuple(words[index: index + state_size])

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

test_set = []
gen = sample(harry_potter, state_size = state_size)

while len(test_set) < test_samples:
    s = next(gen)
    start = s[:state_size]
    if start in text.chain.model:
        test_set.append(s)

text = markovify.Text(open(harry_potter).read(), state_size = state_size)

p_i = []
for t in test_set:
    probs = text.chain.model[tuple(t[:state_size])]
    p = softmax(list(probs.values()))
    a = list(probs.keys())
    pred = np.random.choice(a, p=p)
    p_i.append(dict(zip(a, p))[pred])

perplexity = 2**-(np.sum(np.log2(p_i)) / test_samples)

print "perplexity of state size", state_size, "is", perplexity