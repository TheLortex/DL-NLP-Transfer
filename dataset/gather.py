import os 
from urllib.parse import urlparse
import wget

def setup_brown(file):
    return

def setup_gutenberg(file):
    return

def setup_archive(file):
    return

def setup_shakespeare(file):
    return

def setup_songs(file):
    return

def setup_clinton(file):
    return


datasets = {}
datasets["french"] = {}
datasets["english"] = {}

datasets["english"]["common"] = [("http://www.sls.hawaii.edu/bley-vroman/brown.txt", setup_brown)]
# https://www.gutenberg.org/browse/authors/a#a68
datasets["english"]["austin"] = [("https://www.gutenberg.org/files/31100/31100.txt", setup_gutenberg)]
# https://www.gutenberg.org/browse/authors/d#a37
datasets["english"]["dickens"] = [("https://ia802707.us.archive.org/31/items/thecompleteworks01dickuoft/thecompleteworks01dickuoft_djvu.txt", setup_archive)]
# https://www.gutenberg.org/browse/authors/s#a65
datasets["english"]["shakespeare"] = [("https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt", setup_shakespeare)]
# https://www.gutenberg.org/browse/authors/w#a111
datasets["english"]["wilde"] = [("https://ia600204.us.archive.org/26/items/cu31924103377051/cu31924103377051_djvu.txt", setup_archive)]
# https://www.kaggle.com/mousehead/songlyrics#songdata.csv || Must manually download with a Kaggle account.
datasets["english"]["songs"] = [("/songdata.csv", setup_songs)]
# https://www.kaggle.com/kaggle/hillary-clinton-emails#Emails.csv || Must manually download with a Kaggle account.
datasets["english"]["clinton"] = [("/Emails.csv", setup_clinton)]


def handle_dataset(path, fn):
    return

dir_path = os.path.dirname(os.path.realpath(__file__))

# automated download.
if __name__ == '__main__':
    for language in ["french", "english"]:
        for author, texts in datasets[language].items():
            base_path = dir_path + "/" + language + "/" + author + "/"
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
                    handle_dataset(base_path + file_name, fn)
                    print(author + "|" + file_name + ": Done")
                else:
                    print(author + "|" + file_name + ": OK")