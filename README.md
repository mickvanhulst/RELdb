# RELdb
This package serves the purpose of loading required files for the REL package into a database. Please install it by 
running:

```cmd
pip install git+https://github.com/mickvanhulst/RELdb
```

# Example usage
Usage of this package is mostly integrated in the REL package and in the corresponding tutorials. Nonetheless,
we will briefly go over the functionalities of the package. There are two modes that are of importance for the REL package.
The first is loading loading embedding files (e.g. Wikipedia2Vec) in a sqlite3 database. The second one is loading
Wikipedia p(e|m) dictionaries in a sqlite3 database.


Loading Wikipedia2Vec embeddings into a database is rather straightforward. The implementation differentiates between
either entities or regular words. Entities contain `ENTITY/` in their respective words, while normal words do not. One thing
that we must note here is that we assume a word2vec format for the embedding file.
```python
from RELdb.generic import GenericLookup

save_dir = "C:/test/"

# Embedding load.
emb = GenericLookup('entity_word_embedding', save_dir=save_dir, table_name='embeddings')
emb.load_word2emb('D:/enwiki-20190701-model-w2v-dim300', batch_size=5000, reset=True)

# Query
import torch
embeddings = torch.stack([torch.tensor(e) for e in emb.emb(['in', 'the', 'end'], "embeddings")])
```

Finally, added an example below for loading Wikipedia corpuses into a database. This is fully integrated in the package itself,
but perhaps it can be of service to someone in the future.

```python
from RELdb.generic import GenericLookup

save_dir = "C:/test/"

# Test data
ent_p_e_m_index = {
    "Netherlands": {32796504: 1 / 3, 32796504: 2 / 3},
    "Netherlands2": {32796504: 1 / 3, 32796504: 2 / 3},
}
mention_total_freq = {"Netherlands": 10, "Netherlands2": 100}

# Wiki load.
wiki = GenericLookup('entity_word_embedding', save_dir=save_dir, table_name='wiki',
                     columns={"p_e_m": "blob", "lower": "text", "freq": "INTEGER"})
wiki.load_wiki(ent_p_e_m_index, mention_total_freq, reset=True)

# Query
p_e_m = wiki.wiki("Netherlands", "wiki")
freq = wiki.wiki("Netherlands", "wiki", "freq")
lowercase = wiki.wiki("Netherlands".lower(), "wiki", "lower")

```


# Acknowledgements
This package is based on the [Embeddings](https://github.com/vzhong/embeddings) package by Victor Zhong. It was altered
to not just load embeddings but also work with loading required Wikipedia files. Due to this change we felt the
files were no longer in line with the original package and thus decided to develop a new package. Furthermore, this package will be further developed in the future to deal with
a multitude of REL-related database files and database architectures. 