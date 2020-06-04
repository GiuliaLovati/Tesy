import cv2 as cv
import numpy as np

def imshow_components(image, threshold=70):
    img = cv.threshold(image, 70, 255, cv.THRESH_BINARY)[1]  # ensure binary
    num_labels, labels = cv.connectedComponents(img)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels)) #each label gets a different hue
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch]) #each element of the output array will be a concatenation of the elements of the input arrays

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

    #cv.imshow('labeled.png', labeled_img)
    #cv.waitKey()
    
def connected_components_for_binaryimg(img):
    num_labels, labels = cv.connectedComponents(img)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    #print (blank_ch)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img



#OPERATIONS ON FOUND COMPONENTS:

def equallabels(labels_im, number):   #equal to find 5° column of cv.connectedComponentsWithStats for a specific row (number)
    numlist=[]
    for i in range(labels_im.shape[0]):
        for j in range(labels_im.shape[1]):
            if labels_im[i][j] == number:
                numlist.append(labels_im[i][j])
            else:
                pass
    return len(numlist)


def concompmean(image,thr):     #returns np.mean(stats[:,4])
    lens=[]
    img = cv.threshold(image, thr, 255, cv.THRESH_BINARY)[1]
    num_labels, labels_im = cv.connectedComponents(img)
    for k in range(num_labels):
        newlen = equallabels(labels_im, k)
        lens.append(newlen)
    print (lens)
    return (np.mean(lens))


def selection(image, thr=70):      #selection of connected components with pixel area > certain value (valuemean)
    img = cv.threshold(image, thr, 255, cv.THRESH_BINARY)[1]
    num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(img)
    #print (stats.shape)

    #n° stats rows: n° of connected components
    #5° column stats: number of pixel of that connected component
    #other stats columns describe the box thar contains each component

    areas = stats[:,4]
    areas1 = areas.tolist()
    valuemean = np.mean(areas1)
    print ('Total number of connected components:', len(areas1))
    print ('Average area of connected components:', valuemean)

    bigareasindex = []
    bigareas = []

    for i in areas1:
        if i>=valuemean:
            bigareasindex.append(areas1.index(i))
            bigareas.append(i)

    print ('Labels of connected components with pixel area higher than average:', bigareasindex)  #index 0 : background
    print ('Number of pixels of each selected area:', bigareas) 
    print('')

    bigareasarray = np.array([bigareasindex, bigareas]).T
    print (bigareasarray)
    return bigareasindex, bigareas, bigareasarray


def newimgbigcomponents(image, bigareasindex, thr=70):    #new array image with only the components having area[pixel]> average area of all components
    img = cv.threshold(image, thr, 255, cv.THRESH_BINARY)[1]
    new= np.zeros_like(img,dtype='int32')
    num_labels, labels_im = cv.connectedComponents(img)
    hue = range(0, 255, int(255/len(bigareasindex)))     #set new colors for the selected components in range(0,255)
    for i in range(len(bigareasindex)):       
        #new += np.where(labels_im == bigareasindex[i], labels_im, 0)  #gives problems showing components with label>255
        new += np.where(labels_im == bigareasindex[i], hue[i], 0)    #selected components are mantained with a new label in range(0,255)
        print ('New label for', bigareasindex[i], 'component:', hue[i])
    return new, hue


#FINDING EDGES

def FindingUpperEdges(newimg, huenewimg):
    edges = np.zeros_like(newimg)
    upperlimitx = []
    upperlimity = []
    for i in range(newimg.shape[1]):
        column = newimg[:,i]
        colist = column.tolist()
        for j in huenewimg[1:]:
            try:
                print ('column', i, 'upper edge at:', colist.index(j), ', with label:', j)
                #if in the i-column, pixels with label equal to one of the selected components are present, 
                #it finds the index (row) of the first one with that label
                edges[colist.index(j)][i] = j
                upperlimitx.append(colist.index(j))
                upperlimity.append(i)
            except ValueError:
                pass
    return edges, upperlimitx, upperlimity

def FindingLowerEdges(newimg, huenewimg, edges):
    lowerlimitx = []
    lowerlimity = []
    for i in range(newimg.shape[1]):
        column = newimg[:,i]
        colist = list(reversed(column)) #reversing the column in order to find the last pixel with one of the selected label value
        for j in huenewimg[1:]:
            try:
                print ('column', i, 'lower edge at:', colist.index(j), '(not reversed value), right reversed value:', newimg.shape[0]-colist.index(j), ', with label:', j)
                lowerlimitx.append(newimg.shape[0]-colist.index(j))
                lowerlimity.append(i)
                edges[newimg.shape[0]-colist.index(j)][i] = j #reversing again
            except ValueError:
                pass
    return edges, lowerlimitx, lowerlimity

#THICKNESS CALCULATION

def Thickness(upperlimity, upperlimitx, lowerlimity, lowerlimitx):     #Thickness in pixels
    deltacolumn = np.zeros_like(upperlimity)
    delta = np.zeros_like(upperlimity)
    for i in range(len(upperlimity)):
        for j in range(len(lowerlimity)):
            if i == j:
                delta[i] = lowerlimitx[j] - upperlimitx[i]
                deltacolumn[i] = upperlimity[i]
    return deltacolumn, delta


#Conversion function has 3 possible argument: a thickness values array in pixel for each column of the selected connected components
#Data type specification: automatically US data (important for pixel to second conversion), specify "ITA" for italian data
#Value for dieletric const. : automatically eps = 3.15 from Putzig et al. 2009, tipical of pure water ice. For Grima et al 2009 is 3.1

def Conversion(delta, datatype = "USA", eps = 3.15):
    c = 299792.458  #km/s
    if datatype == "USA":
        convpx = 0.0375*10**(-6) #US data, MROSH_2001: https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=MRO-M-SHARAD-5-RADARGRAM-V1
    elif datatype == "ITA":
        convpx = 0.075*10**(-6) #from 4.3.2.6 TIME ALIGNMENT OF ECHOES paragraph of rdrsis (italian data)
    else:
        print ('uncorrect datatype, try "USA" or "ITA" ')
    deltasec = delta*convpx
    print('Thickness [sec]', deltasec)
    print('Maximum thickness [microsec]', (deltasec*10**6).max())
    deltakm = (deltasec*c)/(2*eps**(0.5)) 
    deltam = deltakm*1000
    print ('Thickness [m]:', deltam)
    print ('Maximum thickness [m]:', deltam.max())
    print ('Average thickness [m]:', deltam.mean())
    return deltasec, deltakm, deltam
