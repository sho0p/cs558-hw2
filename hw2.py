import numpy as np
import cv2
import sys
import math
from PIL import Image
from matplotlib import pyplot as plt

'''
        returns all rgb channels for a given png image
'''
def get_rgb_channels(img_raw):
        imgr = np.zeros(img_raw.size)
        imgg = np.zeros(img_raw.size)
        imgb = np.zeros(img_raw.size)
        for i in range(img_raw.size[0]):
                for j in range(img_raw.size[1]):
                #new_temp = np.asarray([[i[0],i[1]] for i in j])
                        imgr[i][j] = img_raw.getpixel((i,j))[0]
                        imgg[i][j] = img_raw.getpixel((i,j))[1]
                        imgb[i][j] = img_raw.getpixel((i,j))[2]
        #print(new_temp)
        #z = np.hstack((x,y))
        #zr = img_raw_r.ravel()
        #zg = img_raw_g.ravel()
       # zb = img_raw_b.ravel()
        return (imgr, imgg, imgb)

'''
        returns rgb value for a given pixel on a given image
'''
def get_rgb_val(xy, img):
        return (img.getpixel(xy)[0], img.getpixel(xy)[1], img.getpixel(xy)[2])

'''
        selects a random coordinate for a triplet of rgb values
'''
def select_triplet(x, y):
        retx = np.random.randint(0, x)
        rety = np.random.randint(0, y)
        return (retx, rety)

'''
        finds the closest center from a list of centroids, and returns the list of centroids
'''
def find_closest_center(centers, k_test):
        center_list = []
        for indices in k_test:
                min_dist = sys.maxsize
                closest_center = centers[0]
                for center in centers:
                        dist = int(math.sqrt((center[0]-indices[0])**2 + (center[1]-indices[1])**2))
                        if dist < min_dist:
                                min_dist = dist
                                closest_center = center
                center_list.append((indices, closest_center))
        return center_list


'''
        returns the average rgb value between a centroid and its guess

'''
def rgb_quant_val(img, pixset):
        quant_vals = []
        for centroid in pixset:
                r = 0
                g = 0
                b = 0
                for pixel_set in centroid:
                        rgb = get_rgb_val(pixel_set, img)
                        r+=rgb[0]
                        g+=rgb[1]
                        b+=rgb[2]
                r=int(r/(len(centroid)))
                g=int(g/(len(centroid)))
                b=int(b/(len(centroid)))
                quant_vals.append((r,g,b))
        return quant_vals

'''
        creates a [iteration] amount of random centroids
'''
def k_centroid_find(img, iterations):
    k_test = []
    for i in range(iterations):
        k_test.append(select_triplet(img.size[0], img.size[1]))
    return k_test

'''
        treats RGB values as a 3D space and calculates similarity based on distance in that space
'''
def similar_rgb_score(orig_pix, new_val):
        #print(orig_pix)
        #print(new_val)
        return math.sqrt((orig_pix[0]- new_val[0])**2 + (orig_pix[1] - new_val[1])**2 + (orig_pix[2]-new_val[2])**2)


'''
        finds the closest rgb value in the set of quantization values
'''
def find_closest_rgb(orig_pix, quant_vals):
        most_similar = sys.maxsize
        similar_quant_val = None
        for quant_val in quant_vals:
                similarity_score = similar_rgb_score(orig_pix, quant_val)
                if similarity_score < most_similar:
                        most_similar = similarity_score
                        similar_quant_val = quant_val
        return similar_quant_val

'''
        applies quantization values found in find_closest_rgb to the image
'''
def apply_quant(img, quant_vals):
        new_img = np.empty((img.size[0], img.size[1], 3))
        for i in range(img.size[0]):
                for j in range(img.size[1]):
                        orig_pix = get_rgb_val((i,j), img)
                        new_rgb = find_closest_rgb(orig_pix, quant_vals)
                        new_img[i][j][0] = new_rgb[0]
                        new_img[i][j][1] = new_rgb[1]
                        new_img[i][j][2] = new_rgb[2]
        return new_img


'''
        creates the k-means segmentation image. Gets two random sets of centroids,
        calculates the closest pairs, and then quantize forces the value
'''
def k_means_seg(img, iterations=10):
    centroids = []
    for i in range(iterations):
        centroids.append(select_triplet(img.size[0], img.size[1]))
    k_test = k_centroid_find(img, iterations)
    set_k = find_closest_center(centroids, k_test)
    quant_vals = rgb_quant_val(img, set_k)
    new_img = apply_quant(img, quant_vals)
    return new_img

'''
        control for the k_means_seg, basically just so I can comment out a line
        and improve runtime for the SLIC portion
'''
def k_white_tower():
        img_raw = Image.open('white-tower.png')

        new_img=k_means_seg(img_raw)
        new_img = np.rot90(new_img, k=3)
        print(new_img.shape)
        showable_img = Image.fromarray(np.uint8(new_img))
        showable_img.show()


def sectorize(img, sector_size=50):
        

def slic(img):
        sectorize(img)


def slic_wt():
        img = Image.open('wt_slic.png')
        slic(img)


#x = np.random.randint(25,100,25)
#y = np.random.randint(175,255,25)
k_white_tower()

