# Esha Chakrabarty
# 920884665

from scipy import io, sparse, linalg, stats
from scipy.sparse.linalg import svds
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


data_dict = io.loadmat('/Users/eshachakrabarty/Downloads/MAT 167/Digit-Recognition/USPS.mat')

### (a) Load Data
train_patterns = data_dict['train_patterns']
test_patterns = data_dict['test_patterns']

train_labels = data_dict['train_labels']
test_labels = data_dict['test_labels']

## graph first 16 train patterns

fig, axs = plt.subplots(4,4, figsize=(5,5))

for i, ax in enumerate(axs.flatten()):
    img = train_patterns[:,i].reshape(16, 16)
    ax.imshow(img, cmap='plasma')
    ax.set_title(f'Image {i+1}')

plt.tight_layout()
plt.show()

## 0th digit in train patterns represents digit 6
## (i+1) or (7, 0) array indexing -> train_labels[i, j]

train_aves = np.empty([256,10])
for i in range(10):
    digit = train_patterns[:, train_labels[i,:]==1]
    aves = np.mean(digit, axis = 1)
    train_aves[:,i] = aves


fig, axs = plt.subplots(2,5)

for i, ax in enumerate(axs.flatten()):
    img = train_aves[:,i].reshape(16, 16)
    ax.imshow(img, cmap='plasma')
    ax.set_title(f'Avg. of {i}')

plt.show()


## STEP 3
### (a) euclidean distance between test digits and training averages

def euc_dist_perdigit(v1, df):
    res = np.empty([10,])
    for i in range(df.shape[len(df.shape)-1]):
        res[i] = np.sqrt(sum((v1-df[:,i])**2))
    return res

test_classif = np.empty([10, 4649])
for i in range(test_patterns.shape[1]):
    test_classif[:,i] = euc_dist_perdigit(test_patterns[:,i], train_aves)

### (b) find position index of minimum value in each column
    ### this should give us the predicted digit (the index corrs to dig)
test_classif_res = np.argmin(test_classif, axis=0)

### (c) make confusion matrix
# predicted digits where the true values
test = np.argmax(test_labels,axis=0) #test labels by index (gives value)

test_confusion = np.empty([10,10], dtype=int)
for k in range(10):
    predicted = test_classif_res[test_labels[k,:]==1]
    counts = np.bincount(np.sort(predicted), minlength=10)
    test_confusion[:,k]=counts

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=test_confusion)

cm_display.plot(cmap="plasma", values_format="d")
plt.show()

### STEP 4: SVD-based classification 
    ### (a) Find left singular vectors, rank 17
np.random.seed(42)
train_u = np.empty([256,17,10])
for i in range(10):
    cols = train_patterns[:,train_labels[i,:]==1]
    U17, s17, VT17 =svds(cols, k=17)
    train_u[:,:,i] = U17


### (b) expansion coefficients for each test digit
test_svd17 = np.empty([17, 4649, 10])
for k in range(10):
    test_svd17[:,:,k] = train_u[:,:,k].T @ test_patterns

### (c) error between OG test digit img and rank 17 approximation
    ## for k = 0
    ## original test digit image = test_patterns

test_svd17approx = np.empty([256,4649,10])
for i in range(10):
    approx = train_u[:,:,i] @ test_svd17[:,:,i]
    test_svd17approx[:,:,i]=approx


test_svd17res = np.empty([10,4649])
for i in range(test_svd17approx.shape[1]):
    cols = test_svd17approx[:,i,:]
    error = euc_dist_perdigit(test_patterns[:,i], cols)
    test_svd17res[:,i] = error

### (d) create confusion matrix
    ## first find indexes of minimum values
test_svd17_confusion = np.empty([10,10], dtype=int) 

svd17minIndex = np.argmin(test_svd17res,axis=0)

for k in range(10):
    predicted = svd17minIndex[test_labels[k,:]==1]
    counts = np.bincount(np.sort(predicted), minlength=10)
    test_svd17_confusion[:,k]=counts

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=test_svd17_confusion)

cm_display.plot(cmap="plasma", values_format="d")
plt.show()



