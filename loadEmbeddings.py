import numpy as np

def read_embedding(file_name):
    embeddings = {}
    with open(file_name,"r") as f:
        for line in f:
            line = line.replace('\n','').split(' ')
            # print(line[0])
            embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
    return embeddings

if __name__ == "__main__":
    filename = '/Data/WordEmbeddings/glove.6B.50d.txt'
    embeddings = read_embedding(filename)
    print(embeddings['the'])
    print(embeddings['cat'])
