import os 
from urllib.parse import urlparse
import wget
import nltk
import shutil

import csv
nltk.download('punkt')

# One sentence per line, tokens are separated by a space
def standardize(file):
    target = file+".std"


    if os.path.exists(target):
        print ("STD: OK")
        return target

    f = open(file, "r", encoding="latin1")
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
    target_name = file+".txt"
    if os.path.exists(target_name):
        return target_name
    
    f = open(file, "r", encoding="UTF-8")
    f_out = open(target_name, "w", encoding="UTF-8")
    
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for row in reader:
        f_out.write(row["text"])
    
    return target_name

# TODO: extract CSV from zip and generate corpus
def setup_clinton(file):
    target_name = file+".txt"
    if os.path.exists(target_name):
        return target_name
    
    die()
    return target_name


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
# https://www.kaggle.com/mousehead/songlyrics#songdata.csv || Need to setup kaggle API !
datasets["english"]["songs"] = [("https://www.kaggle.com/mousehead/songlyrics#songdata.csv", setup_songs)]
# https://www.kaggle.com/kaggle/hillary-clinton-emails#Emails.csv || Need to setup kaggle API !
# Clinton won't be used right now because it requires a bit more of preprocessing.
#datasets["english"]["clinton"] = [("https://www.kaggle.com/kaggle/hillary-clinton-emails#Emails.csv ", setup_clinton)]

# kaggle datasets download mousehead/songlyrics -f songdata.csv 

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_corpus_location(language, author):
    base_path = dir_path + "/dataset/" + language + "/" + author + "/"
    url, fn = datasets[language][author][0]
    a = urlparse(url)
    file_name = os.path.basename(a.path)
    if a.fragment != "":
        file_name = a.fragment
                    
    path = base_path + file_name
    return standardize(fn(path))

def handle_dataset(path, fn):
    out_file = fn(path)

    return standardize(out_file)

def download(url, target, kaggle):
    if kaggle:
        a = urlparse(url)
        command = "kaggle datasets download "+a.path[1:]+" --unzip -p " + os.path.dirname(target)
        print(command)
        os.system(command)
    else:
        wget.download(url, base_path + file_name)  

def get_datasets(language):
    output = {}
    
    for author in datasets[language].keys():
        output[author] = get_corpus_location(language, author)
        
    return output
        
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
                
                if a.fragment != "":
                    file_name = a.fragment
                    kaggle = True
                else:
                    kaggle = False

                if not os.path.exists(base_path + file_name + ".pkl"):
                    if not os.path.exists(base_path + file_name):
                        print(author + "|" + file_name + ": Downloading")
                        download(url, base_path + file_name, kaggle)

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