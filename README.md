# Word2Vec (CBOW) + FAISS HNSW semantic search

Train word embeddings on a corpus of eight classic novels, visualise them, and find every
word's nearest neighbours with an approximate nearest-neighbour index, then measure how
close the approximate answers are to exact search.

Started as a Deep Learning course project and extended here with an exact-search
evaluation and embedding-space clustering.

![t-SNE projection of the 200 most frequent words](figures/tsne_word_embeddings.png)

## Pipeline

| Step | What happens | Tools |
|---|---|---|
| 1. Corpus | Download 8 public-domain books from Project Gutenberg, strip licence boilerplate, tokenise | `urllib`, `gensim.simple_preprocess` |
| 2. Embeddings | Train a CBOW Word2Vec model (100-d, window 5, min_count 5, 15 epochs) | `gensim` |
| 3. Clusters | KMeans on L2-normalised 100-d vectors, K chosen by cosine silhouette score | `scikit-learn` |
| 4. Visualisation | t-SNE (cosine) projection of the 200 most frequent words, coloured by cluster | `scikit-learn`, `matplotlib`, `adjustText` |
| 5. Search | Top-3 neighbours for every word with an HNSW graph index (M=32, efConstruction=200, efSearch=128), cosine via inner product on unit vectors | `faiss` |
| 6. Evaluation | Recall@3 against exact brute-force search, timing, and an `efSearch` speed/accuracy sweep | `faiss.IndexFlatIP` |

## Corpus

*Crime and Punishment*, *The Brothers Karamazov*, *Moby Dick*, *Dracula*, *Frankenstein*,
*The Trial*, *Alice's Adventures in Wonderland*, *Through the Looking-Glass*: about
6.3M characters, **1,104,241 tokens**, and a **10,298-word** vocabulary after `min_count=5`.
Training takes under a minute on a Colab CPU.

## Results

Sample nearest neighbours found by HNSW (from one training run; multi-threaded Word2Vec training is not
bit-for-bit reproducible, so exact values shift slightly between runs):

| word | 3 nearest neighbours (cosine similarity) |
|---|---|
| six | twelve (0.89), four (0.87), ten (0.87) |
| lawyer | painter (0.75), manufacturer (0.70), businessman (0.66) |
| struck | startled (0.65), realized (0.61), touched (0.60) |

Most similar pairs across the whole vocabulary (each word used at most once):
`von`/`sohn` 0.966, `ii`/`iii` 0.956, `ippolit`/`kirillovitch` 0.951, `moby`/`dick` 0.945,
`per`/`cent` 0.939, `humpty`/`dumpty` 0.934, `mont`/`blanc` 0.934.

These show what CBOW actually learns: words that appear in the *same contexts*. Numbers and
professions group together, but the very highest scores go to names and fixed phrases whose
parts always appear side by side.

### Is the approximate search accurate?

HNSW trades exactness for speed, so the notebook compares it to a brute-force
`IndexFlatIP` over the same vectors and reports **recall@3** (share of the true 3 nearest
neighbours that HNSW also returned), top-1 agreement, query times, and a sweep over
`efSearch`:

![HNSW recall vs. query time](figures/hnsw_tradeoff.png)

At this scale (~10k vectors) exact search is a single matrix multiplication, so HNSW is not
meaningfully faster and can even be slower. Its value is in how it scales: brute-force cost
grows linearly with the number of vectors, HNSW query cost roughly logarithmically.

## Design decisions

- **Cluster in embedding space, not on t-SNE output.** t-SNE keeps local neighbourhoods but
  distorts global distances and cluster sizes, so KMeans on the 2-D coordinates partly
  clusters t-SNE's layout. Clustering happens on the normalised 100-d vectors; t-SNE is only
  used to draw them.
- **Cosine similarity via inner product.** Vectors are L2-normalised once, so FAISS
  inner-product indexes return cosine similarity directly.
- **Fair "top pairs" list.** Words seen fewer than 8 times are dropped (one-off flukes can
  get spuriously high similarity), and each word may appear in only one pair so a single
  tight cluster cannot fill the whole top 10.

## Run it

Open `word2vec_cbow_faiss.ipynb` in Google Colab (Runtime → Run all), or locally:

```bash
pip install -r requirements.txt
jupyter notebook word2vec_cbow_faiss.ipynb
```

The first cell downloads the books from gutenberg.org, so it needs internet access.

## Possible extensions

- Compare CBOW with Skip-gram (`sg=1`) on the same corpus and neighbour lists.
- Benchmark HNSW against brute force on a larger vocabulary (e.g. pretrained GloVe, 400k words)
  where the speed difference becomes visible.
- Add an intrinsic evaluation such as word-similarity benchmarks (SimLex-999, WordSim-353).
