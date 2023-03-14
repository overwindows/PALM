
import os
import random
from tqdm import tqdm
from gensim.corpora import WikiCorpus

if __name__ == '__main__':

    if False:
            inp="/corpus/enwiki-latest-pages-articles.xml.bz2"
            i = 0
            output_file="/corpus/wiki_englist_%07d.txt"%i
            output = open(output_file, 'w',encoding="utf-8")
            wiki = WikiCorpus(inp, lemmatize=None, dictionary={})
            for text in wiki.get_texts():
                output.write(" ".join(text) + "\n")
                i = i + 1
                if (i % 10000 == 0):
                    output.close()

                    output_file = "/corpus/wiki_englist_%07d.txt" % i
                    output = open(output_file, 'w', encoding="utf-8")
                    print("Save "+str(i) + " articles")
            output.close()

    output_dir = '/corpus/wiki/'
    wiki_train_raw = "wiki.train.raw"
    wiki_val_raw = "wiki.valid.raw"
    wiki_test_raw = "wiki.test.raw"

    with open(os.path.join(output_dir, wiki_train_raw), 'w') as fout_wiki_train:
        with open(os.path.join(output_dir, wiki_val_raw), 'w') as fout_wiki_val:
            with open(os.path.join(output_dir, wiki_test_raw), 'w') as fout_wiki_test:

                fileList = os.listdir('/corpus/')
                for file in tqdm(fileList):
                    res = file.split('.')
                    if len(res) == 2:
                        name, ext = res
                    else:
                        continue
                    if ext == 'txt' and name.startswith('wiki'):
                        # print(file)
                        with open(os.path.join('/corpus/', file)) as fin:
                            lines = fin.readlines()
                            # print(len(lines))
                            for line in lines:
                                i = random.randint(1,1000)
                                if i >=1 and i<=800:
                                    fout_wiki_train.write(line)
                                    # fout_wiki_train.write('\n')
                                elif i > 800 and i <= 900:
                                    fout_wiki_val.write(line)
                                    # fout_wiki_val.write('\n')
                                else:
                                    fout_wiki_test.write(line)
                                    # fout_wiki_test.write('\n')
