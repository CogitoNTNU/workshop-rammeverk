import tensorflow as tf

g = tf.add(1, 2, 'KjedeligFunksjonsgraf')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print('result={}'.format(sess.run(g)))
    writer.close()

# tensorboard --logdir=output