import matplotlib
#Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.io as sio
import os
import sys
import numpy
import theano
import theano.tensor as T
import gzip
import cPickle
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


def display(params, digit, epoch, mode = 'mat', size = (56, 56)):

    #epoch contains a list of numbers to show
    #for example, epoch = [0, 2, 4] can show epoch 0 (original stage) and epoch 2 4
    #after running the CNN, params can be used directly, and can also use numpy.load('params.npy') to get
    #digit is a single digit of image set, for example, digit = train_set_x.get_value()[number]
    nkerns=[20, 50]
    rng = numpy.random.RandomState(23455)
    #show original digit
    if os.path.exists('digit') == 0:
        os.mkdir('digit')
    if mode == 'png':
        plt.figure(1)
        plt.gray()
        plt.axis('off')
        plt.imshow(digit.reshape(size))
        plt.savefig('digit/activity of layer0 (original digit).png')
        
    digit = digit.reshape(1, 1, size[0], size[1])
    
    inputdigit = T.tensor4()
    #building CNN with exactly the same parameters
    print '...building layer1'
    layer0_input = inputdigit
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
        image_shape=(1, 1, size[0], size[1]),
        filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))
    
    
    print '...building layer2'
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(1, nkerns[0], (size[0] - 4) / 2, (size[1] - 4) / 2),
        filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))
    
    
    print '...building layer3'
    layer2_input = layer1.output.flatten(2)
    
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * (size[0] / 4 - 3) * (size[1] / 4 - 3),
                         n_out=500, activation=T.tanh)
    
    
    print '...building layer4'
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    
    f = theano.function(inputs = [inputdigit], outputs = [layer0.conv_out, layer0.output, layer1.conv_out, layer1.output, layer2.output, layer3.p_y_given_x, layer3.y_pred])
    
    #export filters and activity in different epochs
    for num in epoch:
        
        print '...epoch ' + str(num)
        
        layer3.W.set_value(params[num][0])
        layer3.b.set_value(params[num][1])
        layer2.W.set_value(params[num][2])
        layer2.b.set_value(params[num][3])
        layer1.W.set_value(params[num][4])
        layer1.b.set_value(params[num][5])
        layer0.W.set_value(params[num][6])
        layer0.b.set_value(params[num][7])
        
        [conv0, output0, conv1, output1, output2, output3, y] = f(digit)
        
        if mode == 'png':
            plt.figure(2)
            plt.gray()
            for i in range(nkerns[0]):
                plt.subplot(4, 5, i + 1)
                plt.axis('off')
                plt.imshow(layer0.W.get_value()[i, 0])
            plt.savefig('digit/filter of layer1 in epoch ' + str(num) + '.png')
            
            plt.figure(3)
            plt.gray()
            for i in range(nkerns[1]):
                plt.subplot(5, 10, i + 1)
                plt.axis('off')
                plt.imshow(layer1.W.get_value()[i, 0])
            plt.savefig('digit/filter of layer2 in epoch ' + str(num) + '.png')
            
            plt.figure(4)
            plt.gray()
            plt.axis('off')
            plt.imshow(layer2.W.get_value())
            plt.savefig('digit/filter of layer3 in epoch ' + str(num) + '.png')
            
            plt.figure(5)
            plt.gray()
            plt.axis('off')
            plt.imshow(layer3.W.get_value())
            plt.savefig('digit/filter of layer4 in epoch ' + str(num) + '.png')
            
            plt.figure(6)
            plt.gray()
            for i in range(nkerns[0]):
                plt.subplot(4, 5, i + 1)
                plt.axis('off')
                plt.imshow(output0[0, i])
            plt.savefig('digit/activity of layer1 after downsampling in epoch ' + str(num) + '.png')
    
            plt.figure(7)
            plt.gray()
            plt.axis('off')
            for i in range(nkerns[1]):
                plt.subplot(5, 10, i + 1)
                plt.axis('off')
                plt.imshow(conv1[0, i])
            plt.savefig('digit/activity of layer2 before downsampling in epoch ' + str(num) + '.png')
    
            plt.figure(8)
            plt.gray()
            plt.axis('off')
            for i in range(nkerns[0]):
                plt.subplot(4, 5, i + 1)
                plt.axis('off')
                plt.imshow(conv0[0, i])
            plt.savefig('digit/activity of layer1 before downsampling in epoch ' + str(num) + '.png')
    
            plt.figure(9)
            plt.gray()
            for i in range(nkerns[1]):
                plt.subplot(5, 10, i + 1)
                plt.axis('off')
                plt.imshow(output1[0, i])
            plt.savefig('digit/activity of layer2 after downsampling in epoch ' + str(num) + '.png')
    
            plt.figure(10)
            plt.gray()
            plt.axis('off')
            plt.imshow(numpy.tile(output2, (10, 1)))
            plt.savefig('digit/activity of layer3 in epoch ' + str(num) + '.png')
    
            plt.figure(11)
            plt.gray()
            plt.axis('off')
            plt.imshow(numpy.tile(output3, (10, 1)))
            plt.savefig('digit/activity of layer4 in epoch ' + str(num) + '.png')

        if mode == 'mat':
            sio.savemat('digit in epoch ' + str(num) + '.mat', {'ActivityOfLayer0' : digit.reshape(size), 
            'ActivityOfLayer1before' : conv0[0],
            'ActivityOfLayer1after' : output0[0],
            'ActivityOfLayer2before' : conv1[0],
            'ActivityOfLayer2after' : output1[0],
            'ActivityOfLayer3' : output2,
            'ActivityOfLayer4' : output3,
            'FilterOfLayer1' : layer0.W.get_value()[:, 0, :, :],
            'FilterOfLayer2' : layer1.W.get_value()[:, 0, :, :],
            'FilterOfLayer3' : layer2.W.get_value(),
            'FilterOfLayer4' : layer3.W.get_value(),
            'y_predict' : y})

    return y

if __name__ == '__main__':
    
    #when using shell, the first parameter is name of digit as .npy format
    #the second and other parameters are the epochs to export
    params = numpy.load('params.npy')
    if sys.argv[1].find('.npy') != -1:
        digit = numpy.load(sys.argv[1])
    elif sys.argv[1].find('.txt') != -1:
        digit = numpy.loadtxt(sys.argv[1])
    size = [int(sys.argv[3]), int(sys.argv[4])]
    epoch = []
    for i in sys.argv[5:]:
        epoch.append(int(i))
    y = display(params, digit, epoch, sys.argv[2])
    print 'classification result of ' + sys.argv[1] + ' is ' + str(y)