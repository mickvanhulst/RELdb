from collections import namedtuple
import zipfile
from numpy import zeros

from base import DB

class GloveEmbedding(DB):
    """
    Reference: http://nlp.stanford.edu/projects/glove
    """

    GloveSetting = namedtuple('GloveSetting', ['url', 'd_embs', 'size', 'description'])
    settings = {
        'common_crawl_48': GloveSetting('http://nlp.stanford.edu/data/glove.42B.300d.zip',
                                        [300], 1917494, '48B token common crawl'),
        'common_crawl_840': GloveSetting('http://nlp.stanford.edu/data/glove.840B.300d.zip',
                                         [300], 2195895, '840B token common crawl'),
        'twitter': GloveSetting('http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                                [25, 50, 100, 200], 1193514, '27B token twitter'),
        'wikipedia_gigaword': GloveSetting('http://nlp.stanford.edu/data/glove.6B.zip',
                                           [50, 100, 200, 300], 400000, '6B token wikipedia 2014 + gigaword 5'),
    }

    def __init__(self, name, save_dir, table_name, columns={"emb": "blob"},
                 d_emb=300):
        """

        Args:
            name: name of the embedding to retrieve.
            d_emb: embedding dimensions.
            show_progress: whether to print progress.
            default: how to embed words that are out of vocabulary. Can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        assert name in self.settings, '{} is not a valid corpus. Valid options: {}'.format(name, self.settings)
        self.setting = self.settings[name]
        assert d_emb in self.setting.d_embs, '{} is not a valid dimension for {}. Valid options: {}'.format(d_emb, name, self.setting)

        self.d_emb = d_emb
        self.name = name
        self.table_name = table_name
        self.columns = columns
        self.save_dir = save_dir
        self.name = name
        path_db = "{}/{}.db".format(save_dir, name)

        self.avg_cnt = {"cnt": 0, "sum": zeros(d_emb)}

        self.db = self.initialize_db(path_db, table_name, columns)

    def emb(self, words, table_name):
        print(words, table_name)
        g = self.lookup(words, table_name)
        return g

    def load_word2emb(self, batch_size=1000):
        self.clear()

        fin_name = self.ensure_file('glove', url=self.setting.url)
        print(fin_name)
        seen = set()
        with zipfile.ZipFile(fin_name) as fin:
            fname_zipped = [fzipped.filename for fzipped in fin.filelist if str(self.d_emb) in fzipped.filename][0]
            with fin.open(fname_zipped, 'r') as fin_zipped:
                batch = []
                for line in fin_zipped:
                    elems = line.decode().rstrip().split()
                    vec = [float(n) for n in elems[-self.d_emb:]]
                    word = ' '.join(elems[:-self.d_emb])
                    if word in seen:
                        continue
                    seen.add(word)
                    batch.append((word, vec))

                    self.avg_cnt["cnt"] += 1
                    self.avg_cnt["sum"] += vec

                    if len(batch) == batch_size:
                        self.insert_batch_emb(batch)
                        batch.clear()

                # Here we are also adding an token based on the average. Take note though that our reported scores
                # for the REL package are based on a random vector as this was also used by Le et al.
                # He reported, however, that he did not notice a difference between using either of the two.
                if self.avg_cnt["cnt"] > 0:
                    batch.append(
                        (
                            "#SND/UNK#",
                            self.avg_cnt["sum"] / self.avg_cnt["cnt"],
                        )
                    )
                    print("Added average for category: #WORD/UNK#")

                if batch:
                    self.insert_batch(batch)

if __name__ == '__main__':
    from time import time
    save_dir = "C:/Users/mickv/Desktop/testemb/"

    emb = GloveEmbedding('common_crawl_48', save_dir=save_dir, table_name='embeddings',
                         columns={"emb": "blob"}, d_emb=300)

    emb.load_word2emb(5000)
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        print(emb.emb([w], 'embeddings'))
        print('took {}s'.format(time() - start))
