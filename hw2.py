import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

#x = np.random.randint(25,100,25)
#y = np.random.randint(175,255,25)
img_raw = Image.open('white-tower.png')
#print(img_raw)
img_raw_r = np.zeros(img_raw.size)
img_raw_g = np.zeros(img_raw.size)
img_raw_b = np.zeros(img_raw.size)
for i in range(img_raw.size[0]):
    for j in range(img_raw.size[1]):
        #new_temp = np.asarray([[i[0],i[1]] for i in j])
        img_raw_r[i][j] = img_raw.getpixel((i,j))[0]
        img_raw_g[i][j] = img_raw.getpixel((i,j))[1]
        img_raw_b[i][j] = img_raw.getpixel((i,j))[2]
#print(new_temp)
#z = np.hstack((x,y))
zr = img_raw_r.ravel()
zg = img_raw_g.ravel()
zb = img_raw_b.ravel()
#z = z.reshape((50,1))
#z = np.float32(z)
plt.hist(zr,256,[0,256]),plt.show(),
plt.hist(zg,256,[0,256]),plt.show(),
plt.hist(zb,256,[0,256]),plt.show()