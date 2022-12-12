import cv2
import numpy as np
from sklearn.cluster import KMeans
import mahotas
from matplotlib import pyplot as plt

ZERNIKE_RADIUS = 50
ZERNIKE_DEGREES = 8

def z_scores(dataset):
    """
    Normalize each feature by Z-Score method (mean=0, std=1)
    """
    sigma = np.std(dataset, axis=0)
    mu = np.mean(dataset, axis=0)
    return (dataset-mu)/sigma

def normalize_img(img):
    """
    Scale img to 2Rx2R keeping aspect-ratio by zero-padding
    """
    radius = ZERNIKE_RADIUS
    h,w = img.shape
    factor = 2*radius/max(h,w)
    # Downscaling
    if factor<1:
        resized = cv2.resize(img, (int(w*factor),int(h*factor)), interpolation = cv2.INTER_AREA)
    # Upscaling
    else:
        resized = cv2.resize(img, (int(w*factor),int(h*factor)), interpolation = cv2.INTER_CUBIC)
    h_pad = int((2*radius-h*factor)/2)
    w_pad = int((2*radius-w*factor)/2)
    normalized = cv2.copyMakeBorder(resized,h_pad,h_pad,w_pad,w_pad,cv2.BORDER_CONSTANT)
    return normalized

def describe(img):
    """
    Extract ortogonal Zernike moments of img with ZERNIKE_RADIUS, ZERNIKE_DEGREES and img centre of mass(default).
    
    References
    ----------
    Teague, MR. (1980). Image Analysis via the General Theory of Moments.  J.
    """
    features = mahotas.features.zernike_moments(img, radius=ZERNIKE_RADIUS, degree=ZERNIKE_DEGREES)
    return features

# ------------------------------------------------------------------------------------------- 
# a - Procesar la imagen de manera de atenuar el ruido y uniformizar el fondo.

# Load Img.
img = cv2.imread('letras_2.tif', cv2.IMREAD_COLOR)
plt.figure()
plt.imshow(img)
plt.show()
cv2.imshow("Input", img)

# Median filter 3x3.
img_median = cv2.medianBlur(img, 3)
cv2.imshow("Filtered", img_median)

# Grayscale.
img_gray = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", img_gray)

# Top-hat.
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
grayscaled = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, se)
cv2.imshow("Top-hat", grayscaled)

# Binary.
_, binary_img = cv2.threshold(grayscaled, 50, 255, cv2.THRESH_BINARY)
binary_img = binary_img.astype(np.uint8)
cv2.imshow("Binary", binary_img)

# Open.
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, se)
cv2.imshow("Binary opened", binary_img)
cv2.waitKey()

# ------------------------------------------------------------------------------------------- 
# b - Procesar  la  imagen  obtenida  en  a.  de  manera  de  segmentar  las  letras,  incorporando  el  bounding 
# box (en  algún  color)  para  cada  letra  en  una  nueva  imagen.

# Connected Components.
n, comps, stats, _ = cv2.connectedComponentsWithStats(binary_img)
bbox_img = cv2.merge([binary_img,binary_img,binary_img])
dataset = list()

for i in range(1,n):
    # Characters detection and bbox.
    obj = (comps == i).astype(np.uint8)
    contours = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = contours[0]
    rect = cv2.minAreaRect(contour)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(bbox_img, [box], 0, (0, 255, 0), 1)

    # Feature extraction.
    x, y, w, h = cv2.boundingRect(contour)
    charr = obj[y:y+h, x:x+w]
    charr_n = normalize_img(charr)
    features = describe(charr_n)
    dataset.append(features)

cv2.imshow('Bboxes',bbox_img)
cv2.waitKey()

# ------------------------------------------------------------------------------------------- 
# c - Implementar  un  algoritmo  que  permita  clasificar  de  manera  automática  las  letras  y  generar  una  
# nueva imagen donde cada clase (i.e., cada letra distinta) esté pintada de diferente color.  

# Clustering.
dataset_n = z_scores(dataset)
clusterer = KMeans(n_clusters=10, max_iter=1000)
clusterer.fit(dataset_n)
labels = clusterer.labels_

# Etiquetar por caracter.
img_chars_labeled = np.zeros_like(comps)
for i in range(1, len(stats)):
    img_chars_labeled[comps == i] = labels[i-1] + 1

# Mapear cada etiqueta a un color diferente separándolas en eje H del espacio HSV.
h = np.uint8(179 * img_chars_labeled / (np.max(img_chars_labeled) + 1))
# S y V a máximo valor, salvo para la componente del fondo.
sv = np.ones_like(h) * 255 * (img_chars_labeled != 0)
# Merge
img_hsv_labeled = cv2.merge([h, sv, sv])

# Mapear a RGB.
img_rgb_labeled = cv2.cvtColor(img_hsv_labeled, cv2.COLOR_HSV2RGB)
cv2.imshow("Labeled", img_rgb_labeled)
cv2.waitKey()