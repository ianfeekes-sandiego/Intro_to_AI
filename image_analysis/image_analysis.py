import numpy as np              #Greyscale conversion
import matplotlib.pyplot as plt #Greyscale conversion
import matplotlib.image as mpimg#Greyscale conversion

#Opens image and returns data structure
def openImage(imgName):
    im = None 
    try: 
        im = mpimg.imread(imgName)
    except:
        print("Image failed to open...")
    return im 

#Converts rgb image into greyscale
def convertImageGreyscale(imgVar):
    return np.dot(imgVar[...,:3], [0.299, 0.587, 0.144])

#Normalizes greyscale image and returns new data structure matrix
def normalizeGreyScale(imgVar):
    ret = imgVar.copy() 
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            ret[i][j]=ret[i][j]/256
    return ret 

#Enhances contrast to the given power
def contrastEnhance(imgVar, powerRaised):
    ret = imgVar.copy() 
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            ret[i][j]=ret[i][j]**powerRaised
    return ret 

#Converts array of image matrices into an array of 1D vectors
def generateVectors(images):
    ret = []
    for i in images: 
        vector = []
        for j in range(len(i)):
            for k in range(len(i[j])):
                vector.append(i[j][k])
        ret.append(vector)
    return ret

#Builts and plots histograms with given parameters
def generateHistograms(vectors, names, figureName, showPlot=False):
    f = plt.figure()
    for i in range(len(vectors)):
        temp = f.add_subplot(1, len(vectors), i+1)
        temp.title.set_text(names[i])
        plt.hist(vectors[i], bins=256)
    plt.savefig(figureName)
    if showPlot: 
        plt.show(block=True)

#Performs binary thresholding on the given image with the given cutoff
def binaryThreshold(imgVar, cutoff):
    ret = imgVar.copy() 
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            if ret[i][j] >= cutoff: ret[i][j]=1
            else: ret[i][j] = 0
    return ret

#Plots and saves the image
def plotImage(imgVar, titleString, imagePlotType, figureName, showPlot=False):
    plt.title(titleString)
    plt.imshow(imgVar, cmap=imagePlotType)
    plt.savefig(figureName)
    if showPlot: 
        plt.show() 

#Plots and saves multiple images into the same plot
def plotImages(images, titles, imagePlotType, figureName, showPlot=False):
    f = plt.figure() 
    for i in range(len(images)):
        temp = f.add_subplot(1, len(images),i+1)
        temp.title.set_text(titles[i])
        plt.imshow(images[i], cmap=imagePlotType)
    plt.savefig(figureName)
    if showPlot:
        plt.show(block=True)

#All parts of assignment
def runMain():
    #1 Open the image
    A = openImage('SanDiego.jpg')
    if A.size==None: exit(1)
    plotImage(A, 'original color image', 'jet', 'SanDiegoOriginal.jpg')
    #2 Convert the image into greyscale
    G = convertImageGreyscale(A)
    plotImage(G, 'grayscale image', 'gray', 'SanDiegoGreyscale.jpg')
    #3 Show the sizes of the matrices of A and G: 
    print(A.shape, " ",A.size)
    print(G.shape, " ",G.size)
    #4 normalize G into C
    C =normalizeGreyScale(G)
    plotImage(C, 'Normalized grayscale', 'gray', 'SanDiegoNormalizedGrayscale.jpg')
    #5a raise pixels to power of .25
    D1 = contrastEnhance(C, 0.25)
    plotImage(D1, 'greyscale image .25 conrast', 'gray', 'SanDiegopt25Greyscale.jpg')
    #5b raise pixels to power of .5, .1, 1.5
    D2 = contrastEnhance(C, .5)
    D3 = contrastEnhance(C, 1)
    D4 = contrastEnhance(C, 1.5)
    #5c create subplots
    images = [D1,D2,D3,D4]
    titles = ["greyscale image .25 contrast", "greyscale image .5 contrast", "greyscale image 1 contrast", "greyscale image 1.5 contrast"]
    plotImages(images, titles, 'gray', "contrast_variation.jpg")
    #5d: differences observed? clear contrast of landmarks and features as contrast increases
    #    and as contrast decreases, things tend to blend together

    #5e/f:  different powers for pixel values until unrecognizable: 6.5 is pretty unrecognizable
    contrastArr = []
    contrastTitles = []
    contrasts = np.arange(9.5, 13.5, 1)
    for i in range(len(contrasts)):
        contrastArr.append(contrastEnhance(C, contrasts[i]))
        contrastTitles.append(str(contrasts[i])+" contrast")
    plotImages(contrastArr, contrastTitles, 'gray', "contrast_adjustment.jpg")
    #6 create histograms for D1-D4
    generateHistograms(generateVectors(images), ["D1","D2","D3","D4"], "Histograms.jpg")
    #7 perform binary thresholding on C with 3 thresholds between 0 and 1 spaced well. 
    threshImgs=[]
    thresholds = np.arange(.25, 1, .25)
    for i in thresholds:
        threshImgs.append(binaryThreshold(C, i))
    plotImages(threshImgs, [".25 threshold", ".5 threshold", ".75 threshold"], 'gray', "thresholdOutput.jpg")
    threshImgs=[]
    thresholds = [.11, .55, .71]
    for i in thresholds:
        threshImgs.append(binaryThreshold(C, i))
    plotImages(threshImgs, [".3 threshold", ".45 threshold", ".6 threshold", ".75 threshold", ".9 threshold"], 'gray', "thresholdOutput.jpg", showPlot=True)

if __name__ == "__main__":
    runMain() 