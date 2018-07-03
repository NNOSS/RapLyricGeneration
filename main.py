# import tensorflow as tf
import loadEmbeddings
import pronouncing

phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B' , 'CH', 'D' , 'DH', 'EH', 'ER', 'EY', 'F' ,
 'G' , 'HH', 'IH', 'IY', 'JH', 'K' , 'L' , 'M' , 'N' , 'NG', 'OW', 'OY', 'P' , 'R' ,
 'S' , 'SH', 'T' , 'TH', 'UH', 'UW', 'V' , 'W' , 'Y' , 'Z' , 'ZH']
 phoneme_dict = {k,v for v,k in enumerate(phoneme_list)}
separation = 3
INPUT_SIZE = 300 + 1 + separation * phoneme_list
OUTPUT_SIZE = 300 + 1 + separation * phoneme_list
HIDDEN_SIZE = 256
BATCH_SIZE = 100
FILENAME = 'kanye_verses.txt'
SAVE_PATH = './Models/GenRap/model.ckpt'
RESTORE = False
LEARNING_RATE = 1e-2
ITERATIONS = 10000

filename = '/Data/WordEmbeddings/glove.6B.300d.txt'
embeddings = loadEmbeddings.read_embedding(filename)
print('Loaded')
words_set = set(embeddings.keys())
print('Set')


def knownWord(word):
    word = word.lower()
    word_l = pronouncing.phones_for_word(word)
    if len(word_l) == 0 or word not in words_set:
        #print(Sinit)
        return False
    return True



def convert_phon(Sinit):
    St1 = Sinit.lower()
    St1 = St1.replace("in'", "ing")
    St1 = pronouncing.phones_for_word(St1)
    myString = St1[0]
    myString = myString.replace("0", "")
    myString = myString.replace("1", "")
    myString = myString.replace("2", "")
    return myString.split(" ")

def read_text(filename):
    with open(filename,'r') as f:
        whole = []
        paragraph = ''
        for line in f:
            if line == "\n":
                whole.append(paragraph)
                paragraph = ''
            else:
                paragraph += line
    return whole

def sanatize_paragraph(paragraph):
    paragraph = paragraph.replace(",", "").replace(".", "").replace("?", "").replace("-", " ").replace("\n", " NEWLINE ")
    paragraph = paragraph.replace('"', '').replace("(","").replace(")","")
    paragraph = paragraph.replace("in'", "ing")
    paragraph = paragraph.split(" ")
    return paragraph

def sanatize_word(word):
    word = word.replace("'", "")
    while not knownWord(word) and len(word):
        word = word[0:-1]
    print(word)

def convert_paragraph(paragraph_list):
    for word in paragraph_list:
        if word == 'NEWLINE':
            pass
        elif knownWord(word):
            phon = convert_phon(word)
        elif word == '':
            pass
        elif 'nigga' in word:
            pass
        else:
            print 'Before: ',word
            sanatize_word(word)

if __name__ == "__main__":
    paragraphs = read_text(FILENAME)
    convert_paragraph(sanatize_paragraph(paragraphs[9]))
    convert_paragraph(sanatize_paragraph(paragraphs[10]))
    convert_paragraph(sanatize_paragraph(paragraphs[11]))
