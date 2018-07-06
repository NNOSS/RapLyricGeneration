# import tensorflow as tf
import loadEmbeddings
import pronouncing
import numpy as np

phoneme_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B' , 'CH', 'D' , 'DH', 'EH', 'ER', 'EY', 'F' ,
 'G' , 'HH', 'IH', 'IY', 'JH', 'K' , 'L' , 'M' , 'N' , 'NG', 'OW', 'OY', 'P' , 'R' ,
 'S' , 'SH', 'T' , 'TH', 'UH', 'UW', 'V' , 'W' , 'Y' , 'Z' , 'ZH']
phoneme_dict = {k:v for v,k in enumerate(phoneme_list)}
SEPARATION = 4
INPUT_EMB_SIZE = 300
INPUT_PHON_SIZE = SEPARATION * len(phoneme_list)
TEMPERATURE  = 1

FILENAME = 'kanye_verses.txt'
SAVE_PATH = './Models/GenRap/model.ckpt'


filename = '/Data/WordEmbeddings/glove.6B.300d.txt'
embeddings = loadEmbeddings.read_embedding(filename)
print('Loaded')
words_set = set(embeddings.keys())
words_list = embeddings.keys()
print('Set')
embeddings_array = np.array([embeddings[word] for word in words_list])
embeddings_array /= np.linalg.norm(embeddings_array, axis = 1)[:,None]
print('match')
hard_words = ['nigga', 'niggas','aingt']
hard_phon = [['N' ,'IH' ,'G' ,'AH'], ['N' ,'IH' ,'G' ,'AH' ,'Z'], ['EY' ,'N' ,'T']]
hard_emb = [embeddings['nigga'],embeddings['niggas'],embeddings['aint']]

def find_word(embedding, num_return = 10):
    arr = find_distances(np.squeeze(embedding))
    # print(arr)
    d = arr.argsort()[(-1*num_return):][::-1]
    # print(d)
    for val in d:
        print(words_list[int(val)])
    print('--------')

def find_distances(embedding):
    return np.mean(embeddings_array * embedding, axis = 1)

def knownWord(word):
    word = word.lower()
    word_l = pronouncing.phones_for_word(word)
    if len(word_l) == 0 or word not in words_set:
        #print(Sinit)
        return False
    return True

def convert_phoneme(phonemes):
    one_hot_length = len(phoneme_list)
    one_hot = np.zeros(SEPARATION * one_hot_length)
    for i, v in enumerate(phonemes[::-1]):
        if i >= SEPARATION-1:
            bump = one_hot_length * (SEPARATION-1)
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
        paragraph = ""
        for line in f:
            if line == "\n":
                whole.append(paragraph)
                paragraph = ""
            else:
                paragraph += line
    return whole

def sanatize_paragraph(paragraph):
    paragraph = paragraph.replace(",", "").replace(".", "").replace("?", "").replace("-", " ").replace("\n", " \n ")
    paragraph = paragraph.replace('"', "").replace("(","").replace(")","")
    paragraph = paragraph.replace("in'", "ing")
    paragraph = paragraph.lower()
    paragraph = paragraph.split(" ")
    return paragraph

def sanatize_word(old_word):
    word = old_word.replace("'", "").replace("\xe2\x80\x99","")
    while not knownWord(word) and len(word):
        word = word[0:-1]
    print(old_word, ' =====> ',word)
    return word

def build_freq_dict(paragraphs):
    freq_dict = {}
    for p in paragraphs:
        p = sanatize_paragraph(p)
        for word in p:
            if word == '\n':
                continue
            if word in hard_words:
                freq_dict[word] = 1 + freq_dict.get(word, 0)
            else:
                if not knownWord(word):
                    word = sanatize_word(word)
                if word == '':
                    continue
                freq_dict[word] = 1 + freq_dict.get(word, 0)
    return {k: 1/(1 + np.sqrt(v/TEMPERATURE)) for k, v in freq_dict.items()}


def convert_paragraph(paragraph_list, freq_dict):
    embedding_list = []
    phoneme_list = []
    EOS_list = []
    weight_list = []
    for word in paragraph_list:
        if word == '\n':
            EOS_list[-1] = 1
            continue

        if word in hard_words:
            index = hard_words.index(word)
            phon = hard_phon[index]
            vec = hard_emb[index]
            weight = freq_dict[word]

        else:
            if not knownWord(word):
                word = sanatize_word(word)
            if word == '':
                continue
            phon = convert_phon(word)
            vec = embeddings[word]
            weight = freq_dict[word]


        embedding_list.append(vec)
        phoneme_list.append(convert_phoneme(phon))
        EOS_list.append(0)
        weight_list.append(weight)
    # for val in [np.array(embedding_list), np.array(phoneme_list), np.array(EOS_list), np.zeros(len(EOS_list))]:
    #     print(val.shape)

    inputs = np.concatenate([np.array(embedding_list), np.array(phoneme_list),
        np.expand_dims(np.array(EOS_list),axis =1),
        np.expand_dims(np.zeros(len(EOS_list)),axis =1)], axis = 1)
    inputs = np.expand_dims(inputs, axis = 1)

    return inputs, np.array(weight_list, dtype=np.float32)


if __name__ == "__main__":
    find_word(embeddings['king'])
    # for val in d:
    #     print(val)
    #     print(words_list[int(val)])

    # paragraphs = read_text(FILENAME)
    # print(paragraphs[9])
    # inputs = convert_paragraph(sanatize_paragraph(paragraphs[9]))
    # print(EOS_list)
    # print(phoneme_list_ex)
    # convert_paragraph(sanatize_paragraph(paragraphs[10]))
    # convert_paragraph(sanatize_paragraph(paragraphs[11]))
