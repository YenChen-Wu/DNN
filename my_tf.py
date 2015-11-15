import time
import numpy as np
import cPickle as pickle
import tensorflow as tf
###############
data_dir = 'data/'
nH = [69,512,48]
batch_size = 128
learning_rate = 0.5
nEpoch = 20
optim = tf.train.MomentumOptimizer    # GradientDescentOptimizer AdagradOptimizer
sess = tf.Session()
###############
def load_data():
  print "loading data for Training set"
  D = pickle.load(open(data_dir+'dev','r'))
  x = D[:,1:]
  y = D[:,0]
  x = (x - x.mean(axis=0))/x.std(axis=0)  # normalization
  return x,y,len(D)/batch_size

def run():
  s = time.time()
  X,Y,nBatch = load_data()

## construction phase
  x = tf.placeholder(tf.float32, shape=(None,69))
  y = tf.placeholder(tf.int32, shape=(None))

  p = feedforward(x,nH)
  loss = get_loss(p,y)
  train_op = optim(learning_rate,0.9).minimize(loss)
  predict_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(p, y, 1), tf.float32))
  init_op = tf.initialize_all_variables()

## execution phase
  sess.run(init_op)
  for epoch in xrange(nEpoch):
    for idx in xrange(nBatch):
      # givens batch
      feed_dict = {x: X[idx * batch_size: (idx + 1) * batch_size],
                   y: Y[idx * batch_size: (idx + 1) * batch_size]}
      _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)   # theano function
    print sess.run(predict_op, feed_dict={x: X, y: Y}) 
  print "\n",time.time()-s," s\n"

def feedforward(x, nH):
  h = [x]
  for i in xrange(len(nH)-1):
    with tf.name_scope('hidden_layer'+str(i)) as scope:
      r = np.sqrt(6.0/(nH[i]+nH[i+1]))
      w = tf.Variable( tf.random_uniform([nH[i], nH[i+1]],-r,r))
      b = tf.Variable( tf.random_uniform([nH[i+1]], -r,r ))
      h.append( tf.nn.relu(tf.matmul(h[i], w) + b) )
  return h[-1]

def get_loss(p, labels):
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense( concated, tf.pack([batch_size, 48]), 1.0, 0.0)
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(p, onehot_labels))

run()
