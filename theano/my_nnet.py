import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
import os,sys,random,time
import argparse
rng = np.random
####################
data_dir = '../data/'
#hidden layer topology
nH = [69,512,48]
batch_size = 128
learning_rate = sys.argv[2] #0.5
lr_decay = 1    # useless..
nEpoch = 20
method = sys.argv[1]
activH = T.nnet.relu
#activ = T.nnet.sigmoid   # T.nnet.ultra_fast_sigmoid    T.nnet.hard_sigmoid()   T.nnet.relu()
#activ = T.tanh          # lr = 0.2
####################
momentum = 0.3 #sys.argv[3]
eps = 0.001
rho = 0.9
beta1 = 0.9
beta2 = 0.999
####################
# TODO
# concatenate  (need to modify io.py)
# shuffle index
# maxout, dropout
# rnn,cnn 
# attetion model
# dqn
####################
def print_info():
    print "Model Structure: ", nH
    print "Batch Size: ", batch_size
    print "Learning Rate: ", learning_rate
    print "Number of Epoch: ", nEpoch
    print "Update Method: ", method
    print "Activation Function: ", activH
    print "Momentum: ", momentum

def load_data(ark):
    # TODO load train ,dev, test once
    s = time.time()
    print "loading data for "+ark+"ing set"
    feat = pickle.load(open(ark,'r'))
    # normalization
    x = feat[:,1:]
    x = (x - x.mean(axis=0))/x.std(axis=0)
    shared_x = theano.shared(np.asarray(x,dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(np.asarray(feat[:,0],dtype=theano.config.floatX),borrow=True)
    print "loaded data for "+ark+"ing set"
    print time.time()-s," s"
    return shared_x, T.cast(shared_y,'int32'), len(feat)/batch_size

class SimpleNet():
    def __init__(self, nL):
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')
        self.w,self.b,self.h = [],[],[x]

        p = self.build_DNN(nL)
        pred = p.argmax(axis = 1)
        nerr = T.mean(T.neq(pred,y))
        cost = -T.mean(T.log(p)[T.arange(y.shape[0]), y])  # cost = T.nnet.categorical_crossentropy(p,y)  mean or sum
        # backpropagation
        self.param = self.w + self.b
        grads = T.grad(cost, self.param)
        updates = my_updates(self.param,grads)
        # theano function
        givens = {x: X[index * batch_size: (index + 1) * batch_size],
                  y: Y[index * batch_size: (index + 1) * batch_size]}
        self.train = theano.function(inputs = [index],outputs = [cost],updates = updates, givens = givens)
        self.predict = theano.function(inputs= [index], outputs= [nerr,pred], givens = givens)

    def build_DNN(self,nL):
        for i in xrange(len(nL)-1):
            r = np.sqrt(6.0/(nL[i]+nL[i+1]))
            self.w.append(theano.shared(np.asarray(rng.uniform(-r,r,(nL[i],nL[i+1])),dtype=theano.config.floatX),name='w'+str(i),borrow=True))
            self.b.append(theano.shared(np.asarray(rng.uniform(-r,r,(nL[i+1])),dtype=theano.config.floatX),name='b'+str(i),borrow=True))
            if i==len(nL)-2:
                return T.nnet.softmax( T.dot(self.h[i], self.w[i]) + self.b[i] )
            self.h.append(activH(T.dot(self.h[i], self.w[i]) + self.b[i]))
        #    if i==len(nL)-2:
        #        activ=T.nnet.softmax
        #    self.h.append(activ(T.dot(self.h[i], self.w[i]) + self.b[i]))
        #return self.h[-1]

def my_updates(param,grads):
    updates = []
    helper = [ theano.shared(p.get_value()*0) for p in param ]
    helper2 = [ theano.shared(p.get_value()*0) for p in param ]
    # learning rate decay
    lr = theano.shared(np.array([learning_rate],dtype=theano.config.floatX)[0])
    updates.append( (lr,lr*lr_decay) )

    if method=='simple':
        updates = [ (p,p-lr*g) for p,g in zip(param,grads) ]
    elif method=='momentum':
        updates.extend( map(lambda _g,g: (_g,_g*momentum - lr*g), helper,grads))		# update uparam
        updates.extend( map(lambda p,_g,g: (p,p+(_g*momentum - lr*g)), param,helper,grads))					# update param
    elif method=='nesterov':
        updates.extend( map(lambda _g,g: (_g,_g*momentum - lr*g), helper,grads))		# update uparam
        updates.extend( map(lambda p,_g,g: (p,p+(_g*momentum**2 - lr*(1+momentum)*g)), param,helper,grads))					# update param
    elif method=='adagrad':
        updates.extend( map(lambda _g,g: (_g,_g + g**2), helper,grads))
        updates.extend( map(lambda p,_g,g: (p,p - lr*g*eps/(eps + (_g + g**2)**0.5)), param,helper,grads))
    elif method=='adadelta': # (no improve)  (no learning rates!)
        ups = [ -(_up+eps)**0.5/( rho*_g + (1-rho)*(g**2) +eps)**0.5*g for _up,_g,g in zip(helper2,helper,grads) ]
        updates.extend( [ (_g, rho*_g + (1-rho)*(g**2)) for _g,g in zip(helper,grads) ])
        updates.extend( [ (_up,rho*_up+ (1-rho)*up**2) for _up,up in zip(helper2,ups) ])
        updates.extend( [ (p,p+up) for p,up in zip(param,ups) ])
    elif method=='rmsprop': # GG   (with momentum?)
        updates.extend( [ (_g, rho*_g + (1-rho)*g) for _g,g in zip(helper,grads) ])
        updates.extend( [ (_g, rho*_g + (1-rho)*(g**2)) for _g,g in zip(helper2,grads) ])
        updates.extend( [ (p,p-lr*g/(_g2-_g**2+eps)**0.5) for p,_g,_g2,g in zip(param,helper,helper2,grads) ])
    elif method=='adam':    # GG
        t = theano.shared(1)    #cast 32?
        updates.extend( [ (t,t+1)] )
        updates.extend( [ (_g, beta1*_g + (1-beta1)*g ) for _g,g in zip(helper,grads) ])      # momentum
        updates.extend( [ (_g, beta2*_g + (1-beta2)*(g**2)) for _g,g in zip(helper2,grads) ]) # adagrad
        m_hat = [ m/(1-beta1**t) for m in helper ]
        v_hat = [ v/(1-beta2**t) for v in helper2 ]
        updates.extend( [ (p,p-lr*mh/((vh)**0.5+eps)) for p,mh,vh in zip(param,helper,helper2) ])

    return updates

def run(nn):
    s = time.time()
    for epoch in xrange(nEpoch):
        # train
        for index in xrange(nBatch):
            nn.train(index)
            sys.stderr.write('\rtraining progress: epoch %d batch %d' %(epoch, index) )
        # validate
        errs = [ nn.predict(index)[0] for index in xrange(nBatch) ]
        print '\taccuracy on training set:', 1 - np.mean(errs)
    # test TODO
    print "\n",time.time()-s," s\n"

#if __name__ == '__main__':
print_info()
X,Y,nBatch = load_data(data_dir+'dev')
nn = SimpleNet(nH)
run(nn)
del X,Y,nBatch,nn
X,Y,nBatch = load_data(data_dir+'train')
nn = SimpleNet(nH)
run(nn)
