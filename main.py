# import tensorflow as tf
import loadEmbeddings
import pronouncing
import numpy as np

phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B' , 'CH', 'D' , 'DH', 'EH', 'ER', 'EY', 'F' ,
 'G' , 'HH', 'IH', 'IY', 'JH', 'K' , 'L' , 'M' , 'N' , 'NG', 'OW', 'OY', 'P' , 'R' ,
 'S' , 'SH', 'T' , 'TH', 'UH', 'UW', 'V' , 'W' , 'Y' , 'Z' , 'ZH']
phoneme_dict = {k:v for v,k in enumerate(phoneme_list)}
separation = 4
INPUT_EMB_SIZE = 200
INPUT_PHON_SIZE = separation * phoneme_list
OUTPUT_EMB_SIZE = 200
OUTPUT_PHON_SIZE = separation * phoneme_list
HIDDEN_SIZE = 256
BATCH_SIZE = 100
FILENAME = 'kanye_verses.txt'
SAVE_PATH = './Models/GenRap/model.ckpt'
RESTORE = False
LEARNING_RATE = 1e-2
ITERATIONS = 10000

filename = '/Data/WordEmbeddings/glove.6B.200d.txt'
embeddings = loadEmbeddings.read_embedding(filename)
print('Loaded')
words_set = set(embeddings.keys())
print('Set')
hard_words = ['nigga', 'niggas','aingt']
hard_phon = [['N' ,'IH' ,'G' ,'AH'], ['N' ,'IH' ,'G' ,'AH' ,'Z'], ['EY' ,'N' ,'T']]
hard_emb = [embeddings['nigga'],embeddings['niggas'],embeddings['aint']]


def knownWord(word):
    word = word.lower()
    word_l = pronouncing.phones_for_word(word)
    if len(word_l) == 0 or word not in words_set:
        #print(Sinit)
        return False
    return True

def convert_phoneme(phonemes):
    one_hot_length = len(phoneme_list)
    one_hot = np.zeros(separation * one_hot_length)
    for i, v in enumerate(phonemes[::-1]):
        if i >= separation-1:
            bump = one_hot_length * (separation-1)
            one_hot[phoneme_dict[v] + bump] = 1
        else:
            bump = one_hot_length * i
            one_hot[phoneme_dict[v] + bump] = 1
    return one_hot

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
    paragraph = paragraph.replace(",", "").replace(".", "").replace("?", "").replace("-", " ").replace("\n", " \n ")
    paragraph = paragraph.replace('"', '').replace("(","").replace(")","")
    paragraph = paragraph.replace("in'", "ing")
    paragraph = paragraph.lower()
    paragraph = paragraph.split(" ")
    return paragraph

def sanatize_word(word):
    word = word.replace("'", "")
    while not knownWord(word) and len(word):
        word = word[0:-1]
    print(word)
    return word

def convert_paragraph(paragraph_list):
    embedding_list = []
    phoneme_list = []
    EOS_list = []
    for word in paragraph_list:
        if word == '\n':
            EOS_list[-1] = 1
            continue

        if word in hard_words:
            index = hard_words.index(word)
            phon = hard_phon[index]
            vec = hard_emb[index]
        else:
            if not knownWord(word):
                word = sanatize_word(word)
            if word == '':
                continue
            phon = convert_phon(word)
            vec = embeddings[word]

        embedding_list.append(vec)
        phoneme_list.append(convert_phoneme(phon))
        EOS_list.append(0)
    return embedding_list, phoneme_list, EOS_list


if __name__ == "__main__":

    paragraphs = read_text(FILENAME)
    print(paragraphs[9])
    embedding_list_ex, phoneme_list_ex, EOS_list = convert_paragraph(sanatize_paragraph(paragraphs[9]))
    print(EOS_list)
    print(phoneme_list_ex)
    # convert_paragraph(sanatize_paragraph(paragraphs[10]))
    # convert_paragraph(sanatize_paragraph(paragraphs[11]))
