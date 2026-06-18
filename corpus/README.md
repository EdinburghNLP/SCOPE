# Corpus Setup

The retrieval server requires a Wikipedia corpus and a FAISS index.

## Required Files

- `wiki-18.jsonl` — Wikipedia corpus (~14 GB)
- `e5_Flat.index` — FAISS flat index built with `intfloat/e5-base-v2` embeddings (~61 GB)

## Download

These files are too large to include in the repository. Contact the authors for access or reconstruct from source using the index builder:

```bash
python search/index_builder.py \
    --retrieval_method e5 \
    --model_path intfloat/e5-base-v2 \
    --corpus_path corpus/wiki-18.jsonl \
    --save_dir corpus
```

This writes `corpus/e5_Flat.index`, which is the default index path used by
the training and evaluation scripts.
