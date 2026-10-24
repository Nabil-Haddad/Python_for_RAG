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

---

## Contextual embeddings

*(To be written.)*
