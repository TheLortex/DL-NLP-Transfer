import os 
from urllib.parse import urlparse
import wget
import nltk
import shutil

nltk.download('punkt')

# One sentence per line, tokens are separated by a space
def standardize(file):
    target = file+".std"

    f = open(file, "r", encoding="latin1")

    if os.path.exists(target):
        print ("STD: OK")
        return target

    f_out = open(target, "w", encoding="UTF-8")

    buffer = ""
    
    line = f.readline()
    n_lines = 0
    n_paragraphs = 0
    n_sentences = 0

    while line:
        n_lines += 1
        buffer += " " + line
        if line == "\n":
            n_paragraphs += 1

            sentences = nltk.sent_tokenize(buffer)

            for sent in sentences:
                n_sentences += 1
                words = nltk.word_tokenize(sent)
                output_sent = " ".join(words)
                f_out.write(output_sent + "\n")

            buffer = ""
            

        line = f.readline()
    print(n_lines, n_paragraphs, n_sentences)
    return file+".std"

def no_setup(file):
    return file

# TODO: extract CSV from zip and generate corpus
def setup_songs(file):
    return file

# TODO: extract CSV from zip and generate corpus
def setup_clinton(file):
    return file


datasets = {}
datasets["french"] = {}
datasets["english"] = {}

datasets["english"]["common"] = [("http://www.sls.hawaii.edu/bley-vroman/brown.txt", no_setup)]
# https://www.gutenberg.org/browse/authors/a#a68
datasets["english"]["austin"] = [("https://www.gutenberg.org/files/31100/31100.txt", no_setup)]
# https://www.gutenberg.org/browse/authors/d#a37
datasets["english"]["dickens"] = [("https://ia802707.us.archive.org/31/items/thecompleteworks01dickuoft/thecompleteworks01dickuoft_djvu.txt", no_setup)]
# https://www.gutenberg.org/browse/authors/s#a65
datasets["english"]["shakespeare"] = [("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt", no_setup)]
# https://www.gutenberg.org/browse/authors/w#a111
datasets["english"]["wilde"] = [("https://ia600204.us.archive.org/26/items/cu31924103377051/cu31924103377051_djvu.txt", no_setup)]
# https://www.kaggle.com/mousehead/songlyrics#songdata.csv || Must manually download with a Kaggle account.
#datasets["english"]["songs"] = [("/songdata.csv.zip", setup_songs)]
# https://www.kaggle.com/kaggle/hillary-clinton-emails#Emails.csv || Must manually download with a Kaggle account.
#datasets["english"]["clinton"] = [("/Emails.csv.zip", setup_clinton)]


dir_path = os.path.dirname(os.path.realpath(__file__))

def get_corpus_location(language, author):
    base_path = dir_path + "/dataset/" + language + "/" + author + "/"
    url, fn = datasets[language][author][0]
    a = urlparse(url)
    file_name = os.path.basename(a.path)
    path = base_path + file_name
    return standardize(fn(path))

def handle_dataset(path, fn):
    out_file = fn(path)

    return standardize(out_file)


# automated download.
if __name__ == '__main__':
    files = []
    for language in ["french", "english"]:
        for author, texts in datasets[language].items():
            base_path = dir_path + "/dataset/" + language + "/" + author + "/"
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            for (url, fn) in texts:
                a = urlparse(url)
                file_name = os.path.basename(a.path)

                if not os.path.exists(base_path + file_name + ".pkl"):
                    if not os.path.exists(base_path + file_name) and url[0] != '/':
                        print(author + "|" + file_name + ": Downloading")
                        wget.download(url, base_path + file_name)  

                    print(author + "|" + file_name + ": Preprocessing")
                    files.append(handle_dataset(base_path + file_name, fn))
                    print(author + "|" + file_name + ": Done")
                else:
                    print(author + "|" + file_name + ": OK")
    
    # Concat everything for the corpus
    final_path = dir_path + "/dataset/" + language + "/corpus.txt"
    with open(final_path,'wb') as wfd:
        for f in files:
            if f != None:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd, 1024*1024*10)