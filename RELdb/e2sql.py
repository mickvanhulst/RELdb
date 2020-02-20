from REL_database.base_embedding import Embedding
from numpy import zeros, float32 as REAL
import numpy as np
from gensim import utils

class GenericLookup(Embedding):
    def __init__(
        self,
        name,
        save_dir,
        file_name=None,
        dict_file=None,
        table_name="embeddings",
        d_emb=300,
        categories=[],
        default="none",
        reset=False,
        batch_size=5000,
        columns={"emb": "blob"},
    ):
        """
        Args:
            name: name of the embedding to retrieve.
            d_emb: embedding dimensions.
            show_progress: whether to print progress.
            default: how to embed words that are out of vocabulary. Can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        self.file_name = file_name
        self.dict_file = dict_file

        self.avg_cnt = {"word": {"cnt": 0, "sum": zeros(d_emb)}}
        for c in categories:
            self.avg_cnt[c] = {"cnt": 0, "sum": zeros(d_emb)}

        # self.setting = self.settings[name]
        assert default in {"none", "random", "zero"}

        path_db = "{}/{}.db".format(save_dir, name)

        self.categories = categories
        self.reset = reset
        self.d_emb = d_emb
        self.name = name
        self.db = self.initialize_db(
            path_db, table_name, columns
        )  # self.path(path.join('{}:{}.db'.format(name, d_emb))))
        self.default = default
        self.batch_size = batch_size
        self.table_name = table_name
        self.columns = columns

    def emb(self, words, table_name):
        g = self.lookup(words, table_name)
        return g

    # TODO:
    def wiki(self, mention, table_name, column_name="p_e_m"):
        g = self.lookup_wik(mention, table_name, column_name)
        return g

    def load_word2emb(self, categories):
        unicode_errors = "strict"
        encoding = "utf-8"
        # fin_name = self.ensure_file(path.join('glove', '{}.zip'.format(self.name)), url=self.setting.url)
        batch = []
        start = time()

        # Loop over file.
        with utils.open(self.file_name, "rb") as fin:
            # Determine size file.
            header = utils.to_unicode(fin.readline(), encoding="utf-8")
            vocab_size, vector_size = (
                int(x) for x in header.split()
            )  # throws for invalid file format

            if self.limit:
                vocab_size = min(vocab_size, self.limit)

            for line_no in range(vocab_size):
                line = fin.readline()
                if line == b"":
                    raise EOFError(
                        "unexpected end of input; is count incorrect or file otherwise damaged?"
                    )

                parts = utils.to_unicode(
                    line.rstrip(), encoding=encoding, errors=unicode_errors
                ).split(" ")

                if len(parts) != vector_size + 1:
                    raise ValueError(
                        "invalid vector on line %s (is this really the text format?)"
                        % line_no
                    )

                word, vec = parts[0], np.array([REAL(x) for x in parts[1:]])

                if word in self.seen:
                    continue

                self.seen.add(word)
                batch.append((word, vec))

                if "ENTITY/" in word:
                    self.avg_cnt["entity"]["cnt"] += 1
                    self.avg_cnt["entity"]["sum"] += vec
                else:
                    self.avg_cnt["word"]["cnt"] += 1
                    self.avg_cnt["word"]["sum"] += vec

                if len(batch) == self.batch_size:
                    print("Another {}".format(self.batch_size), line_no, time() - start)
                    start = time()
                    self.insert_batch(batch)
                    batch.clear()

        for c in self.categories:
            if self.avg_cnt[c]["cnt"] > 0:
                batch.append(
                    (
                        "#{}UNK#".format(c),
                        self.avg_cnt[c]["sum"] / self.avg_cnt[c]["cnt"],
                    )
                )
                print("Added average for category: #{}UNK#".format(c))

        if self.avg_cnt["word"]["cnt"] > 0:
            batch.append(
                (
                    "#WORD/UNK#",
                    self.avg_cnt["word"]["sum"] / self.avg_cnt["word"]["cnt"],
                )
            )
            print("Added average for category: #WORD/UNK#")

        if batch:
            self.insert_batch(batch)
        self.create_index()


if __name__ == "__main__":
    from time import time

    base_url = "/mnt/c/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/wiki_2020/"
    ent_p_e_m_index = {
        "Netherlands": {32796504: 1 / 3, 32796504: 2 / 3},
        "Netherlands2": {32796504: 1 / 3, 32796504: 2 / 3},
    }

    mention_total_freq = {"Netherlands": 10, "Netherlands2": 100}

    start = time()
    save_dir = "/mnt/c/Users/mickv/Google Drive/projects/entity_tagging/deep-ed/data/wiki_2014//generated/embeddings/"

    # wiki or embeddings
    emb = GenericLookup('common_crawl_48', save_dir=save_dir, table_name='wiki',
                         columns={"p_e_m": "blob", "lower": "text", "freq": "INTEGER"},
                         d_emb=300)

    emb.load_word2emb(categories=["ENTITY/"])
    # categories=["ENTITY/"]
    start = time()
    print(emb.wiki("Netherlands", "wiki"))
    print(emb.wiki("Netherlands", "wiki", "freq"))
    print(emb.wiki("Netherlands".lower(), "wiki", "lower"))

    print(time() - start)
