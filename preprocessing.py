import numpy
#function frame can move digit into big frame with certain size
#offset is the random offset of digit, offset = 0 means digit will be located
#right in the center of frame
def frame(digit, size = (56, 56), offset = 0):    
    f = numpy.zeros(size)
    x = size[0] / 2 - 14 + numpy.random.randint(- offset, offset + 1)
    y = size[1] / 2 - 14 + numpy.random.randint(- offset, offset + 1)
    f[x:x + 28][:, y:y + 28] = digit.reshape(28, 28)
    return f.reshape(1, -1)

#function addblock will add random black block to a digit
def addblock(digit, size = (4, 4)):
    x = numpy.random.randint(0, 28 - size[0])
    y = numpy.random.randint(0, 28 - size[1])
    digit[x:x + size[0]][:, y:y + size[1]] = 0