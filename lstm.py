import tensorflow as tf
import transformData
SEPARATION = 4
PHONEME_LIST_SIZE = 39
INPUT_EMB_SIZE = 300
INPUT_PHON_SIZE = SEPARATION * PHONEME_LIST_SIZE
EOS_SIZE = 1
EOP_SIZE = 1
NUM_WORDS = 5455
OUTPUT_SIZE = NUM_WORDS + INPUT_PHON_SIZE + EOS_SIZE + EOP_SIZE
HIDDEN_SIZE = 512
FILENAME = 'kanye_verses.txt'

RESTORE = False
LEARNING_RATE = 1e-2
ITERATIONS = 20000
SUMMARY_FILEPATH = '/Models/RapGen/Summaries/'

sequence = tf.placeholder(tf.float32, [None, 1, OUTPUT_SIZE], name = 'input')
word_weights = tf.placeholder(tf.float32, [None], name = 'word_weights')
correct_word = tf.placeholder(tf.float32, [None], name = 'word_weights')
correct_word_one_hot = tf.one_hot(correct_word)
input_seq = sequence[:-1]
input_seq = tf.concat([tf.zeros([1,1,OUTPUT_SIZE]), input_seq], axis = 0)
output, state = tf.nn.dynamic_rnn(
    tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, num_proj = OUTPUT_SIZE,name = 'lstm_cell'),
    input_seq,
    dtype=tf.float32,
)
word_vec_label, other_label = tf.split(sequence, [NUM_WORDS, OUTPUT_SIZE - NUM_WORDS],axis = 2)
word_vec_output, other_output = tf.split(output, [NUM_WORDS, OUTPUT_SIZE - NUM_WORDS],axis = 2)

word_vec_loss_vec= tf.losses.softmax_cross_entropy_with_logits(word_vec_label,word_vec_output)
other_loss_vec= tf.nn.sigmoid_cross_entropy_with_logits(labels=other_label,
    logits=other_output)

word_vec_loss= tf.reduce_mean(word_vec_loss_vec)
other_loss= tf.reduce_mean(other_loss_vec)

word_vec_weighted_loss= tf.reduce_mean(word_vec_loss_vec * word_weights)



word_vec_loss_summary=tf.summary.scalar('Word Vec Loss', word_vec_loss)
other_loss_summary=tf.summary.scalar('Other Loss', other_loss)
train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(word_vec_weighted_loss + other_loss)

if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)

    paragraphs = transformData.read_text(FILENAME)
    i = 0

    freq_dict= transformData.build_freq_dict(paragraphs)
    {word:i for i, word in enumerate(freq_dict.keys())}

    while i < ITERATIONS:
        print('EPOCH')
        for p in paragraphs:
            # print(p)
            if p in ['', ' ', '\n']:
                continue
            inputs, weight_list = transformData.convert_paragraph(transformData.sanatize_paragraph(p), freq_dict)
            feed_dict = {sequence: inputs, word_weights: weight_list}
            word_vec_loss_sum, other_loss_sum, _, word_vec_output_ex = sess.run([word_vec_loss_summary,other_loss_summary,train, word_vec_output], feed_dict = feed_dict)
            train_writer.add_summary(word_vec_loss_sum, i)
            train_writer.add_summary(other_loss_sum, i)
            i += 1

            if i > 10000:
                print(p)
                list_p = transformData.sanatize_paragraph(p)
                list_p.remove('')
                for j, vec in enumerate(word_vec_output_ex):
                    # print(vec)
                    print(list_p[j])
                    print('--------')
                    transformData.find_word(vec)
        paragraphs = transformData.read_text(FILENAME)
