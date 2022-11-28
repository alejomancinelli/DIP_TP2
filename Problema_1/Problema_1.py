import cv2
from matplotlib import pyplot as plt
import numpy as np

# ------------------------------------------------------------------------------------------- 
# a - Procesar la imagen de manera de atenuar el ruido y uniformizar el fondo.  

# Cargar imagen
img_og = cv2.imread('letras_2.tif')
# Filtro de mediana en los canales RGB
img_filtered = np.empty_like(img_og)
for i in range(0,3):
    img_aux = cv2.medianBlur(img_og[:,:,i], 5)
    img_filtered[:,:,i] = img_aux.copy()
# RGB a escala de grises
img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
# Se aplica top-hat
img_gray_uniforme = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
# Imagen de palabras en blanco y negro
_, img_bw = cv2.threshold(img_gray_uniforme, 50, 255, cv2.THRESH_BINARY)

# Se muestran los resultados
# plt.figure()  # Por si no anda el cv2.imshow()
# plt.title("Imagen original")
# plt.imshow(cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))
# plt.show()
cv2.imshow("Imagen original", cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB))
cv2.imshow("Filtro de mediana", img_gray)
cv2.imshow("Fondo uniforme", img_gray_uniforme)
plt.figure()
plt.title("Imagen en blanco y negro")
plt.imshow(img_bw, cmap = 'gray')
plt.show()

# ------------------------------------------------------------------------------------------- 
# b - Procesar  la  imagen  obtenida  en  a.  de  manera  de  segmentar  las  letras,  incorporando  el  bounding box  
# (en  algún  color)  para  cada  letra  en  una  nueva  imagen

# Se juntan las letras para separar las palabras
img_close = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
cv2.imshow("Mascara de palabras", img_close)
# Cantidad de palabras a partir de los componentes conectados
num_labels, labels = cv2.connectedComponents(img_close)
plt.figure()
plt.imshow(labels, cmap = 'jet') 
plt.show()
new_img_bw = np.empty_like(img_bw)
new_img_bw_borders = np.empty_like(img_bw)
# Relación de aspecto de las letras
letter_rel = 0.9

for m in range(1,num_labels):
    # Se muestra solamente una palabra
    word_mask = (labels == m).astype(np.uint8)
    word_mask[word_mask != 0] = 1
    img_word = img_bw * word_mask
    # Se limpian un poco las letras
    word_clean = cv2.morphologyEx(img_word, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    # cv2.imshow("Palabras", word_clean)
    # A partir de la relación de aspecto de las letras, se aproxima la cantidad de letras por palabra    
    word_contour, _ = cv2.findContours(word_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    word_rect = cv2.minAreaRect(word_contour[0]) 
    (cx, cy), (w, h), _ = word_rect
    word_heigth = min((w, h)) 
    word_width =  max((w, h))
    letters_aprox_count = round(word_width / (letter_rel * word_heigth))
    # Cantidad de letras a partir de los elementos conectados
    num_let, _ = cv2.connectedComponents(word_clean)
    # print("Aprox: {} - Real: {}".format(letters_aprox_count, num_let-1))
    # Si la cantidad de letras es menor a la cantidad de letras estimada, es que hay letras conectadas y se las debe separar
    while(num_let-1 < letters_aprox_count):
        word_clean = cv2.erode(word_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        num_let, _ = cv2.connectedComponents(word_clean)
    # Se cargan las letras separadas en una nueva imagen
    new_img_bw = new_img_bw + word_clean
cv2.imshow("Imagen en limpio", new_img_bw)

new_img_bw_borders = new_img_bw.copy()
# Imagen de escala de grises a RGB para mostrar los bordes en otro color
img_letters_borders = cv2.cvtColor(new_img_bw_borders, cv2.COLOR_GRAY2RGB)
# Contornos de las letras
letter_contours, _ = cv2.findContours(new_img_bw_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
# Se dibujan el bounding box de cada letra
for contour in letter_contours:
    rect = cv2.minAreaRect(contour) 
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_letters_borders, [box], 0, (255, 0, 0), thickness=2)
cv2.imshow("Letras con bounding box", img_letters_borders)

# ------------------------------------------------------------------------------------------- 
# c - Implementar  un  algoritmo  que  permita  clasificar  de  manera  automática  las  letras  y  generar  una  
# nueva imagen donde cada clase (i.e., cada letra distinta) esté pintada de diferente color.

size = np.size(new_img_bw)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

img_esqueleto = np.empty_like(new_img_bw)
for k in range(1, num_labels):
    done = False
    skel = np.zeros(new_img_bw.shape, np.uint8)
    # Se muestra solamente una palabra
    word_mask = (labels == k).astype(np.uint8)
    word_mask[word_mask != 0] = 1
    img_word = new_img_bw * word_mask
    cv2.imshow("Palabra", img_word)
    while(not done):
        eroded = cv2.erode(img_word, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_word, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_word = eroded.copy()
    
        zeros = size - cv2.countNonZero(img_word)
        if zeros==size:
            done = True
    cv2.imshow("Esqueleto", skel)        
    img_esqueleto = img_esqueleto + skel
# plt.figure()
# plt.imshow(img_esqueleto, cmap = 'gray')
# plt.show()

esqueleto_close = cv2.morphologyEx(img_esqueleto, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
# plt.figure()
# plt.imshow(esqueleto_close, cmap = 'gray')
# plt.show()

esqueleto_close_rgb = cv2.cvtColor(esqueleto_close, cv2.COLOR_GRAY2RGB)
lines = cv2.HoughLines(esqueleto_close_rgb, rho=1, theta=np.pi/180, threshold=250)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)   # https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]        
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv2.line(esqueleto_close_rgb,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('hough lines', esqueleto_close_rgb)
plt.figure()
plt.imshow(esqueleto_close_rgb)
plt.show()