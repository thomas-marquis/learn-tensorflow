import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

#%%

constant = tf.constant([1, 2, 3], name="toto")
constant = tf.Print(constant, [constant])

#%%

with tf.Session() as sess:
    tensor = constant * constant
    tensor = tf.Print(tensor, [tensor])
    tensor.eval(session=sess)


#%%
zeros = tf.zeros([3, 3])
print(zeros)


#%%
mat = [[23.0, -45.0], [1.0, 3.0]]
op = tf.matmul(mat, tf.random_uniform([2, 2]))

with tf.Session() as sess:
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    metadata = tf.RunMetadata()

    sess.run(op, options=options, run_metadata=metadata)

    print(metadata.partition_graphs)
    print(metadata.step_stats)

#%%
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./", sess.graph)