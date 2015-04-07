import numpy
import theano
import theano.tensor as T
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import scipy.optimize as so


def Buildnet(params, nkerns = [20, 50], batch_size = 500):

    rng = numpy.random.RandomState(23455)

    datasets = load_data(0)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (28, 28)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=3)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
	
    f = theano.function(inputs = [index], outputs = [layer2.output, layer3.y_pred, y], 
		givens = {
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]})
   
#    numepoch = len(params)
    layer3.W.set_value(params[-1][0])
    layer3.b.set_value(params[-1][1])
    layer2.W.set_value(params[-1][2])
    layer2.b.set_value(params[-1][3])
    layer1.W.set_value(params[-1][4])
    layer1.b.set_value(params[-1][5])
    layer0.W.set_value(params[-1][6])
    layer0.b.set_value(params[-1][7])
    
    outputvectors = numpy.zeros((10000, 500))
    labels = numpy.zeros((10000, 1))
    reallabels = numpy.zeros((10000, 1))
    
    
    
    for minibatch_index in xrange(n_test_batches):
    
        vector, label, reallabel = f(minibatch_index)
    
        outputvectors[minibatch_index * batch_size:(minibatch_index + 1) * batch_size] = vector
        labels[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, 0] = label
        reallabels[minibatch_index * batch_size:(minibatch_index + 1) * batch_size, 0] = reallabel
    
    return [outputvectors, labels, reallabels]

def perturb_random(params, shape, oldoutput, nkerns = [20, 50], batch_size = 500):
    
    print '... building the model'
    rng = numpy.random.RandomState(23455)

    x = T.tensor4()
    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)


	
    f = theano.function(inputs = [x], outputs = [layer2.output, layer3.y_pred])
    
    layer3.W.set_value(params[-1][0])
    layer3.b.set_value(params[-1][1])
    layer2.W.set_value(params[-1][2])
    layer2.b.set_value(params[-1][3])
    layer1.W.set_value(params[-1][4])
    layer1.b.set_value(params[-1][5])
    layer0.W.set_value(params[-1][6])
    layer0.b.set_value(params[-1][7])
    
#    perturb 500 shapes at each iteration, with ptimes iterations
    perturbed = numpy.tile(shape[0], (500, 1))
    oldoutputs = numpy.tile(oldoutput, (500, 1))
    label = shape[1]
    ptimes = 500
    imagelength = numpy.sqrt(numpy.sum(shape[0] ** 2))
    outputlength = numpy.sqrt(numpy.sum(oldoutput ** 2))
    p = []
    s = []
    for i in range(ptimes):
        print 'perturbing ' + str(i) + ' ......'
        perturbation = numpy.random.normal(0, 0.15, perturbed.shape)
        perturblength = numpy.sqrt(numpy.sum(perturbation ** 2, axis = 1))
        shapes = perturbed + perturbation
        outputs, labels = f(shapes.reshape(500, 1, 28, 28))
        distances = numpy.sum((outputs - oldoutputs) ** 2, axis = 1)
        pos = numpy.argmax(distances)
        print 'distance ' + str(numpy.sqrt(distances[pos]))
        pert = {}
        pert['perturbation'] = perturbation[pos]
        pert['plength'] = perturblength[pos]
        pert['ilength'] = imagelength
        pert['olength'] = outputlength
        pert['distance'] = numpy.sqrt(distances[pos])
        pert['output'] = outputs[pos]
        pert['label'] = labels[pos]
        p.append(pert)
        if len(numpy.nonzero(labels != label)[0]) != 0:
            print 'success!' + str(label) + ' '
            pos = numpy.nonzero(labels != label)[0][0]
            print labels[pos]
            pert = {}
            pert['perturbation'] = perturbation[pos]
            pert['plength'] = perturblength[pos]
            pert['ilength'] = imagelength
            pert['olength'] = outputlength
            pert['distance'] = numpy.sqrt(distances[pos])
            pert['output'] = outputs[pos]
            pert['label'] = labels[pos]
            s.append(pert)
    return p, s
def perturb_bfgs(perturbation, params, shape, oldoutput, c = 1, nkerns = [20, 50], batch_size = 1):
    
    #print '... building the model'
    rng = numpy.random.RandomState(23455)

    x = T.tensor4()
    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=3)

	
    f = theano.function(inputs = [x], outputs = [layer2.output, layer3.y_pred])
    
    layer3.W.set_value(params[-1][0])
    layer3.b.set_value(params[-1][1])
    layer2.W.set_value(params[-1][2])
    layer2.b.set_value(params[-1][3])
    layer1.W.set_value(params[-1][4])
    layer1.b.set_value(params[-1][5])
    layer0.W.set_value(params[-1][6])
    layer0.b.set_value(params[-1][7])
    

    perturbed = shape
    oldoutputs = oldoutput
    distances = 0
    perturblength = numpy.sqrt(numpy.sum(perturbation ** 2))
    shapes = perturbed + perturbation
    outputs, labels = f(shapes.reshape(1, 1, 28, 28))
    print labels
    for o in oldoutputs:
        distances += numpy.sqrt(numpy.sum((outputs - o) ** 2))
    distances /= len(oldoutputs)
    return c * perturblength + distances
    
def print_resnorm(x):
    print "residual norm = ", fun(x)
    
if __name__ == '__main__':
    params = numpy.load('params.npy')
    #testoutputs = Buildnet(params)
    #numpy.save('outputs.npy', testoutputs)
    test_set = numpy.load('test_set.npy')
    clist = [1.0, 1.5, 2.0]
    for s in [4, 5, 6, 8, 9]:
        testoutputs = numpy.load('output' + str(s) + '.npy')
        for c in clist:
            fun = lambda x:perturb_bfgs(x, params, test_set[0][0], testoutputs, c)
            res = so.minimize(fun, numpy.random.normal(0, 0.1, 784), method = 'L-BFGS-B', 
                    options = {'maxiter':100, 'disp':True},
                    callback = print_resnorm)
            numpy.save(str(s) +'res' + str(c) + '.npy', res)
