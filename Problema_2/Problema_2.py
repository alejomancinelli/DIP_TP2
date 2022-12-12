import cv2
from matplotlib import pyplot as plt
import numpy as np

RHO_TH = 0.88

# Muesta la imagen, sea color o escala de grises
def mostrar_imagen(img_src, color_img=False):
    if(color_img == False):
        plt.imshow(img_src, cmap = 'gray')
    else:
        plt.imshow(img_src)
    plt.show()

# Rellena el contorno de una imagen binaria
def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)  
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 

# ------------------------------------------------------------------------------------------- 
# a - Procesar la imagen de manera de segmentar las monedas y los dados de manera automática.

# Cargar imagen
img_og = cv2.imread('monedas.jpg')
img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)

# RGB a escala de grises
img_gray = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)

# Uniformización del fondo mediante filtro de mediana
img_fondo_uniforme = cv2.medianBlur(img_gray, 7)

# Esto anda masomenos pero capaz no es la idea. Faltaría ajustar mejor los parámetrosa
# detected_circles = cv2.HoughCircles(img_fondo_uniforme, cv2.HOUGH_GRADIENT, 1, 200, param1 = 100, param2 = 15, minRadius = 1, maxRadius = 250)

# Detección de bordes
img_bordes = cv2.Canny(img_fondo_uniforme, 0, 100)

# Dilatación para cerrar agujeros
img_dilatada = cv2.dilate(img_bordes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))

# Relleno de los objetos
img_rellena = fillhole(img_dilatada)

# Limpieza
erode = cv2.erode(img_rellena, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
img_clean = cv2.morphologyEx(erode, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (151,151)))

# Detección de objetos
num_labels, labels = cv2.connectedComponents(img_clean)
mask_objetos = np.empty_like(img_og)

for i in range(1, num_labels):
    # Analisis de un objeto por vez
    mask = (labels == i).astype(np.uint8)
    obj = img_clean * mask
    contour, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    # Relación entre área y perímetro
    rho = 4 * np.pi * (cv2.contourArea(contour[0]) / (cv2.arcLength(contour[0], True) ** 2))

    # Si es mayor a 0.88 es un círculo (moneda), sino otra cosa (dado)
    if(rho > RHO_TH):
        mask_objetos[:,:,0] = mask_objetos[:,:,0] + obj
    else:
        mask_objetos[:,:,2] = mask_objetos[:,:,2] + obj

# Highlight en rojo de las monedas y en azul de los dados
alpha = 0.5
img_resaltada = cv2.addWeighted(mask_objetos, alpha, img_og, 1 - alpha, 0)

mostrar_imagen(img_og, color_img=True)
mostrar_imagen(img_gray)
mostrar_imagen(img_fondo_uniforme)
mostrar_imagen(img_bordes)
mostrar_imagen(img_dilatada)
mostrar_imagen(img_rellena)
mostrar_imagen(img_clean)
mostrar_imagen(img_resaltada, color_img=True)

# ------------------------------------------------------------------------------------------- 
# b - Clasificar los distintos tipos de monedas y realizar un conteo, de manera automatica. 

# Máscara solamente de monedas
mask_monedas = mask_objetos[:,:,0]
mask_monedas[mask_monedas != 0] = 1
img_solo_monedas = img_fondo_uniforme * mask_monedas

# Se guardan las áreas para la clasificación
num_labels, labels = cv2.connectedComponents(img_solo_monedas)
areas = []

for i in range(1, num_labels):
    mask = (labels == i).astype(np.uint8)
    obj = img_solo_monedas * mask
    con, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    areas.append(cv2.contourArea(con[0]))

# Clasificación mediante K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # criteria --> ( type, max_iter, epsilon )
ret, moneda_label, center = cv2.kmeans(np.array(areas).astype(np.float32), 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  

# Se pintan las monedas en distintos colores según su valor
cant_monedas = [0,0,0]
mask_tipo_moneda = np.empty_like(img_og)

for i in range(1, num_labels):
    n_label = moneda_label[i-1][0]
    cant_monedas[n_label] = cant_monedas[n_label] + 1 
    mask = (labels == i).astype(np.uint8)
    obj = mask_monedas * mask
    obj[obj != 0] = 255

    if(n_label == 0):
        mask_tipo_moneda[:,:,0] = mask_tipo_moneda[:,:,0] + obj
        continue 
    if(n_label == 1):
        mask_tipo_moneda[:,:,1] = mask_tipo_moneda[:,:,1] + obj
        continue 
    if(n_label == 2):
        mask_tipo_moneda[:,:,2] = mask_tipo_moneda[:,:,2] + obj
        continue

alpha = 0.5
img_monedas_por_valor = cv2.addWeighted(mask_tipo_moneda, alpha, img_og, 1 - alpha, 0)

print("Cantidad de monedas por tipo:\nRojas - {}\nVerdes- {}\nAzules- {}".format(cant_monedas[0], cant_monedas[1], cant_monedas[2]))

mostrar_imagen(img_solo_monedas)
mostrar_imagen(mask_tipo_moneda, color_img=True) 
mostrar_imagen(img_monedas_por_valor, color_img=True) 

# ------------------------------------------------------------------------------------------- 
# c - Determinar el número que presenta cada dado mediante procesamiento automático.

# Creación de máscara para los dados
mask_dados = mask_objetos[:,:,2]
mask_dados[mask_dados != 0] = 1
img_solo_dados = img_fondo_uniforme * mask_dados

# Imagen a blanco y negro (invertida) para obtener nomás los valores de los dados
_, img_dados_th = cv2.threshold(img_solo_dados, 100, 255, cv2.THRESH_BINARY_INV)

# Se dejan nomás los valores de los dados filtrando por area
num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(img_dados_th)
img_solo_valores = img_dados_th.copy()

for i in range(num_labels):
    if(stats[i, cv2.CC_STAT_AREA] > 5000):
        img_solo_valores[labels == i] = 0
    else:
        img_solo_valores[labels == i] = 255

# Mejora de los valores
img_valores_close = cv2.morphologyEx(img_solo_valores, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

# Se dejan nomás los valores que son un círculo, a partir de la relación área-perímetro
num_labels, labels = cv2.connectedComponents(img_valores_close)
img_valores_clean = img_valores_close.copy()

for i in range(2,num_labels):
    mask = (labels == i).astype(np.uint8)
    obj = img_valores_close * mask
    con, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    rho = 4 * np.pi * (cv2.contourArea(con[0]) / (cv2.arcLength(con[0], True) ** 2))

    if(rho < RHO_TH):
        img_valores_clean[labels == i] = 0

# Viendo un dado por vez, se cuentan los círculos que representan su valor
img_dados_con_valores = img_og.copy()
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dados)

for i in range(1,num_labels):
    mask = (labels == i).astype(np.uint8)
    vals = img_valores_clean * mask
    val_dado, _ = cv2.connectedComponents(vals)
    print(val_dado - 1)
    cv2.putText(img_dados_con_valores, text=str(val_dado-1), org=(stats[i, cv2.CC_STAT_LEFT]+5, stats[i, cv2.CC_STAT_TOP]+5), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=5, color=(255, 0, 0), thickness=10)

mostrar_imagen(img_solo_dados)
mostrar_imagen(img_dados_th)
mostrar_imagen(img_solo_valores)
mostrar_imagen(img_valores_close)
mostrar_imagen(img_valores_clean)
mostrar_imagen(img_dados_con_valores)