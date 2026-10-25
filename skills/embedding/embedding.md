# Embeddings

An embedding turns a piece of text (a word, a sentence, a document) into a vector of numbers, so that texts with a similar meaning end up close to each other in vector space.

There are two main types:

| | Static | Contextual |
|---|---|---|
| One vector per | word (type) | word occurrence (token) |
| Depends on the sentence | No | Yes |
| "bank" in "river bank" vs "bank account" | Same vector | Different vectors |
| Examples | Word2Vec, GloVe, FastText | BERT, sentence-transformers, OpenAI embeddings |
| Cost | Very cheap (a lookup table) | Heavier (a neural network forward pass) |

---

## Static embeddings

### What does it produce?

A lookup table: one fixed vector per word in the vocabulary.

```python
kv = api.load("glove-wiki-gigaword-50")
kv["king"]   # float32 array, shape (50,)
len(kv)      # 400000 words, each with its own 50-dimension vector
```

- The vector never changes, whatever the sentence around the word is.
- Words that appear in similar contexts get similar vectors (`king` is close to `queen`).
- A word that is not in the vocabulary has no vector (`"Bambouzeling" in kv` is `False`).
- To embed a sentence, you have to combine the word vectors yourself (for example, by averaging them).

### How was it trained?

The idea behind it: *a word is known by the company it keeps*. Words used in similar contexts should get similar vectors.

- **Word2Vec** (skip-gram): slide a window over a large corpus. For each word, predict the words around it. The vectors are the weights that are learned to make those predictions good. Negative sampling is used: real (word, neighbour) pairs are pushed together, and random pairs are pushed apart.
- **GloVe** (the one used here): first count how often every pair of words appears together in the corpus (a co-occurrence matrix). Then learn vectors so that the dot product of two word vectors matches the log of their co-occurrence count:

  `w_i · w_j + b_i + b_j ≈ log(X_ij)`

In both cases the model starts with random vectors and adjusts them until words that share contexts end up close to each other. The `glove-wiki-gigaword-50` model was trained on Wikipedia and Gigaword (6 billion tokens), with 50 dimensions.

### Dot product vs cosine similarity

Both measure how close two vectors are, but they treat the length of the vectors differently.

- **Dot product**: `a · b = |a| |b| cos(θ)`. It depends on the angle between the vectors **and** on their lengths.
- **Cosine similarity**: `a · b / (|a| |b|)`. It depends only on the angle. The length is removed.

If the vectors are normalised to unit length (see the `norm=True` option of `get_vector`), the two are identical, because `|a| = |b| = 1`.

**During training** (dot product):
- Word2Vec and GloVe use the dot product as their score.
- The length of a vector is free to change, and it ends up carrying information. Frequent words tend to get different lengths than rare ones.
- The length can be used to encode things such as how often a word appears or how confident the model is.

**During training with cosine similarity**:
- Length no longer matters, only direction does. Every word is forced onto a sphere, so the vectors can only encode meaning through direction.
- The score is limited to the range [-1, 1], so it is usually scaled with a temperature to get usable gradients.
- This is the common choice for models trained with contrastive learning, such as sentence embedding models.

**At search time** (comparing two words or documents), cosine similarity is the usual choice. A very frequent word can have a large vector, and with the dot product it would look similar to everything. Cosine compares only the meaning, not the size.

| | Dot product | Cosine similarity |
|---|---|---|
| Uses vector length | Yes | No |
| Range | Unbounded | [-1, 1] |
| Same as the other if vectors are unit length | Yes | Yes |
| Typical use | Training score (Word2Vec, GloVe) | Comparing vectors, contrastive training |

### Worked example

Two small 2D vectors, `A = (4, 3)` and `B = (1, 1)`:

- Dot product: `4*1 + 3*1 = 7`
- Lengths: `|A| = 5`, `|B| = √2 ≈ 1.414`
- Cosine similarity: `7 / (5 * 1.414) ≈ 0.99`, the two arrows point almost the same way.

Now make B ten times longer, `B = (10, 10)`. The direction is unchanged:

- Dot product: `4*10 + 3*10 = 70` (ten times bigger)
- Cosine similarity: still `≈ 0.99`

The dot product grows with the length of the vectors, the cosine does not. This is the "length vs direction" difference in one example. The code and the plot are in [vectors_similarity.py](vectors_similarity.py).

### Dimensions

The `50` in `glove-wiki-gigaword-50` is the number of values in each vector. More dimensions (GloVe also has 100, 200 and 300) can capture more nuance, but the model is bigger and comparisons are slower. The vocabulary size (400,000 words) is a separate number: it is the number of vectors in the table.

### Nearest words and word arithmetic

Since similar words are close, we can ask for the nearest neighbours of a word:

```python
kv.most_similar("cat", topn=5)   # [(word, cosine score), ...]
```

Directions in the space also carry meaning, so vectors can be added and subtracted:

```python
# king - man + woman
kv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
```

`most_similar` builds `king - man + woman`, then returns the words with the highest cosine similarity to that result. The input words are excluded from the results. With this model `queen` comes out at or near the top: removing "man" from "king" leaves the idea of royalty, and adding "woman" gives the female version. It does not always work, the results depend on the training corpus.

### How to read the 2D plot

A vector with 50 values cannot be drawn, so `plot_vectors` in [static_embedding.py](static_embedding.py) projects two words onto the 2D plane that contains both of them. The first word lies along the x-axis, and the second word is split into a part along that axis and a part at 90 degrees to it. The lengths and the angle between the arrows are the real ones: the smaller the angle, the higher the cosine similarity.

PCA is not suitable for this. It subtracts the mean first, and with only two points the mean is their midpoint, so the two words always come out as exact opposites, whatever they are.

### Limits of static embeddings

- **One vector per word:** "bank" has the same vector in "river bank" and "bank account". All meanings are mixed in one vector (polysemy).
- **Out-of-vocabulary words:** a word that is not in the table (a typo, a new word, a rare name) has no vector. The model also only has lowercase words.
- **No sentence meaning:** a sentence has to be built by combining word vectors yourself, and word order is lost if you average them.
- **Bias:** the vectors reflect the text they were trained on, including its stereotypes.

These limits are the reason for contextual embeddings.

---

## Contextual embeddings

*(To be written.)*
