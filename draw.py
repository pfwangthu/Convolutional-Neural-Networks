import numpy as np
import matplotlib.pyplot as plt

def MyPlot(matrix, size = 28):
    plt.gray()
    plt.axis('off')
    plt.imshow(matrix.reshape(size, size))

#generate any kind of regular or irregular polygons with random sizes and centers
def DrawPolygon(num, size, pos = 1, direction = 1, ratio = 1, pattern = 0):
    if direction == 1:
        phi = np.random.random() * 2 * np.pi
    else:
        phi = 0
    c = zip([np.cos(phi + i * 2 * np.pi / num) for i in range(num)], 
             [np.sin(phi + i * 2 * np.pi / num) for i in range(num)])
    matrix = np.zeros((28, 28))
    if pattern == 1:
        background = np.random.random((28, 28))
    else:
        background = np.ones((28, 28))
    edges = []
    if pos == 1:
        center = (np.random.randint(size, 28 - size),
              np.random.randint(int(np.ceil(ratio * size)), 28 - int(np.ceil(size))))
    else:
        center = (14, 14)         
    cx = [center[0] + size * x[0] for x in c]
    cy = [min(27, center[1] + ratio * size * y[1]) for y in c]
    cx.append(cx[0])
    cy.append(cy[0])
    for i in range(num):
        if cy[i] < cy[i + 1]:
            ymin = cy[i]
            ymax = cy[i + 1]
            xmin = cx[i]
            xmax = cx[i + 1]
        else:
            ymin = cy[i + 1]
            ymax = cy[i]
            xmin = cx[i + 1]
            xmax = cx[i]
        edges.append((ymin, ymax, xmin, xmax))
    
    for i in range(num):
        matrix[round(cx[i]), round(cy[i])] = 1
    for i in range(num):
        (ymin, ymax, xmin, xmax) = edges[i]
        if (ymax - ymin) >= (xmax - xmin) and (ymax - ymin) >= (xmin - xmax):
            m = (xmax - xmin) / (ymax - ymin)
            for y in range(int(np.round(ymin)), int(np.ceil(ymax))):
                x = np.round(xmin + m * (y - ymin))
                matrix[x][y] = 1
        else:
            k = (ymax - ymin) / (xmax - xmin)
            for x in range(int(np.round(min(xmin, xmax))), int(np.ceil(max(xmin, xmax)))):
                y = np.round(ymin + k * (x - xmin))
                matrix[x][y] = 1
    for x in range(28):
        flag = 0
        start = -1
        end = 0
        for y in range(1, 28):
            if((matrix[x][y - 1] == 0 and matrix[x][y] == 1 and flag == 0) or (matrix[x][0] == 1 and flag == 0)):
                flag = 1
                start = y
            elif(matrix[x][y - 1] == 0 and matrix[x][y] == 1 and flag == 1):
                flag += 1
                end = y
        if flag == 2:
            matrix[x][start:end] = background[x][start:end]
    matrix.shape = 28 * 28
    return matrix


#generate a circle in a matrix of (28, 28), size is the diameter of circle
def DrawCircle(size):
    radius = size / 2
    matrix = np.zeros(784)
    matrix.shape = 28, -1
    x = np.random.randint(radius, 28 - radius)
    y = np.random.randint(radius, 28 - radius)
    for i in range(28):
        for j in range(28):
            if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                matrix[i][j] = 1
    matrix.shape = 784
    return matrix
#generate a rectangle in a matrix, with 2 directions
def DrawRect(size):
    matrix = np.zeros(784)
    matrix.shape = 28, -1
    x = np.random.randint(0, 28 - size[0])
    y = np.random.randint(0, 28 - size[1])
    k = np.random.randint(0, 2)
    if k == 0:
        matrix[x:x + size[0]][:, y:y + size[1]] = 1
    else:
        matrix[y:y + size[1]][:, x:x + size[0]] = 1
    matrix.shape = 784
    return matrix
#generate a triangle in a matrix, with 4 directions
def DrawTriangle(size):
    matrix = np.zeros(784)
    matrix.shape = 28, -1
    x = np.random.randint(0, 28 - size)
    y = np.random.randint(0, 28 - size)
    k = np.random.randint(0, 4)
    if k == 0:
        for i in range(x, x + size):
            matrix[i][y:y + size - (i - x)] = 1
    elif k == 1:
        for i in range(x, x + size):
            matrix[i][y + (i - x):y + size] = 1
    elif k == 2:
        for i in range(x, x + size):
            matrix[i][y:y + (i - x)] = 1
    else:
        for i in range(x, x + size):
            matrix[i][y + size - (i - x):y + size] = 1
    matrix.shape = 784    
    return matrix

#old function to generate shapes
def GenerateSets(num, size):
    Sets = []
    Labels = []
    minsize = size[0:2]
    maxsize = size[2:]
    for i in range(num):
        k = np.random.randint(1, 4)
        newsize = [np.random.randint(minsize[0], maxsize[0]),
                   np.random.randint(minsize[1], maxsize[1])]
        if k == 1:
            Sets.append(DrawCircle(newsize[0]))
            Labels.append(1)
        if k == 2:
            Sets.append(DrawRect(newsize))
            Labels.append(2)
        if k == 3:
            Sets.append(DrawTriangle(newsize[0]))
            Labels.append(3)
    Sets = [Sets, Labels]
    return Sets
    
#new function to generate shapes
def GeneratePolygons(edges, num, size, pos = 1, direction = 1, changeratio = [1], addpattern = 0):
    Sets = []
    Labels = []
    minsize = size[0]
    maxsize = size[1]
    for i in range(num):
        NO = np.random.randint(0, len(edges))
        numedge = edges[NO]
        thisratio = np.random.choice(changeratio)
        thissize = np.random.randint(minsize, maxsize)
        Sets.append(DrawPolygon(numedge, thissize, pos, direction, thisratio, addpattern))
        Labels.append(NO)
    return [Sets, Labels]
