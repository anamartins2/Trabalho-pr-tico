import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import regionprops

def sementes_dados (nome_arquivo):
    img_bgr = cv2.imread(nome_arquivo, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)
    r_bilateral = cv2.bilateralFilter(r, 15, 15, 55)
    #hist_r_bilateral = cv2.calcHist([r_bilateral], [0], None, [256], [0, 256])
    l_f, img_l_f = cv2.threshold(r_bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_segmentada = cv2.bitwise_and(img_rgb, img_rgb, mask=img_l_f)
    mascara = img_l_f.copy()  # cópia pra não modificar a mascara original
    cnts, h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ng = len(cnts)
    sementes = np.zeros((ng, 1))
    #dimensao = np.zeros((ng, 1))
    eixo_menor = np.zeros((ng, 1))
    eixo_maior = np.zeros((ng, 1))
    razao = np.zeros((ng, 1))
    area_1 = np.zeros((ng, 1))
    perimetro = np.zeros((ng, 1))


    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)  # cria retangulos em volta do contorno x = posição inicio no eixo x, y = inicio no eixo y; w=largura;h=altura
        obj = img_l_f[y:y + h, x:x + w]  # recortando os graos
        obj_rgb = img_segmentada[y:y + h, x:x + w]
        obj_bgr = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('graos/s' + str(i + 1) + '.png', obj_bgr)
        cv2.imwrite('graos/sb'+str(i+1)+'.png',obj)
        # regionprops solicita a imagem binária
        regiao = regionprops(obj)  # https: // scikit - image.org / docs / dev / api / skimage.measure.html  # skimage.measure.regionprops
        #print('Semente: ', str(i + 1))
        sementes[i, 0] = i + 1
        #print('Dimensão da Imagem: ', np.shape(obj))
        # dim = np.shape(obj)
        # dimensao[0, i] = dim
        #print('Medidas Físicas')
        #print('Centroide: ', regiao[0].centroid)
        #print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
        eixo_menor[i, 0] = regiao[0].minor_axis_length
        #print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
        eixo_maior[i, 0] = regiao[0].major_axis_length
        #print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length)
        razao[i, 0] = eixo_maior[i, 0] / eixo_menor[i, 0]
        # contourArea solicita o contorno
        area = cv2.contourArea(c)
        area_1[i, 0] = area
        perimetro[i, 0] = cv2.arcLength(c, True)
    data = np.concatenate((sementes, eixo_menor, eixo_maior, razao, area_1, perimetro), axis=1)
    df = pd.DataFrame(data, columns=['Semente', 'Eixo Menor', 'Eixo Maior', 'Razao', 'Area', 'Perimetro'])
    df.set_index('Semente', inplace=True)
    return df,img_l_f,img_segmentada, cnts

dados,img_l_f,img_segmentada, cnts = sementes_dados('316.ge')
print(dados)
dados.to_csv('tabela316.csv')
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    obj = img_l_f[y:y + h, x:x + w]  # recortando os graos
    obj_rgb = img_segmentada[y:y + h, x:x + w]
    obj_bgr = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite('graos105/s' + str(i + 1) + '.png', obj_bgr)
    cv2.imwrite('graos105/sb' + str(i + 1) + '.png', obj)