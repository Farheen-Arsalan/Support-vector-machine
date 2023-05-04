from sklearn import datasets,metrics,svm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = datasets.load_digits()
# 1797 samples 
# each image is 8x8 
# feature is 1x64 
# real Mnist digit dataset - > 
# 60000 samples
# 28x28 image 
# 1x784 feature 
# subplot 
plt.figure(1)
for i in range(1,13,1):
    plt.subplot(3,4,i)
    sampleNo = np.random.randint(0,1796,size=(1))
    im = data.images[sampleNo[0],:,:] # accesing the first image
    plt.imshow(im,cmap = 'gray')
    plt.axis('off')
    plt.title('sample: '+str(data.target[sampleNo[0]]))


