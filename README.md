# README

## Convolución Actividad
**Mariana Islas Mondragón / A01253435**  
Dentro de este codigo se aplica un kernel de sobel a una imagen utilizando convolución.

Para poder ejecutar este codigo es necesario descargar las librerias numpy, cv2 y matplotlib. También es necesario revisar la dirección de la imagen cat que se utiliza dentro del codigo, si no es correcta actualizar dirección correcta en linea 45. Después de esto es posible utilizar el programa dandole click a ejecutar. Se desplegara una ventana con la imagen original y la imagen con el filtro aplicado.

---

## Padding Actividad 
**Mariana Islas Mondragón / A01253435**  
Dentro de este codigo se aplica un kernel de sobel a una imagen utilizando convolución y se le agrega padding.

Para poder ejecutar este codigo es necesario descargar las librerías numpy, cv2 y matplotlib. También es necesario revisar la dirección de la imagen cat que se utiliza dentro del codigo, si no es correcta actualizar dirección correcta en linea 46. Después de esto es posible utilizar el programa dandole click a ejecutar. Se desplegara una ventana con la imagen original, la imagen original con padding y la imagen con el filtro aplicado y padding.

## Proyecto Equipo
# MANUAL, INVESTIGACIÓN Y PROPUESTA

### 20 de Marzo de 2025

**Maria Regina Orduño Lopez - A01252959**  
**Mariana Islas Mondragón - A01253435**  
**Mariana Carrillo Holguin - A01253358**  
**Zuleyca Guadalupe Balles Soto - A01741687**  

## MANUAL
Para ejecutar este código, es necesario tener instaladas las siguientes bibliotecas de Python:
- NumPy
- OpenCV
- Matplotlib

Además, hay que verificar que la imagen esté guardada, así como también verificar que el path de esta sea correcto.

## INTRODUCCIÓN
En este repositorio se realizó un proyecto en donde se tenía como objetivo crear un filtro nuevo utilizando distintos filtros. Para poder realizar esto se utilizaron kernels y convoluciones para poder combinar las matrices de la imagen y el kernel seleccionado.

El proceso de convolución se refiere a una operación matemática en donde se combinan dos matrices. Esta operación se ejecuta sobre cada píxel de la imagen para así poder obtener una imagen nueva con el kernel aplicado. El proceso de la operación matemática es el siguiente: se aplica la matriz del kernel sobre la matriz original y se realiza una multiplicación de matrices, en donde se multiplica celda con su celda correspondiente y finalmente se suman todos los resultados para poder obtener el primer valor de la matriz resultante. Se va recorriendo columna por columna hasta llegar al final de las columnas y después se saltará a la siguiente fila para repetir el mismo proceso. Finalmente, se obtendrá la matriz resultante en donde cada celda se refiere a la intensidad de cada píxel de nuestra nueva imagen con el filtro aplicado.

## FILTRO GAUSSIANO
El filtro Gaussiano se refiere a un kernel que se utiliza para reducir detalles y ruido en las imágenes, así como para difuminarlas, disminuyendo la transición abrupta entre píxeles. En este sentido, es similar al filtro promedio, pero emplea un kernel diferente que sigue la forma de una joroba gaussiana (con forma de campana). Este operador realiza un promedio ponderado de los píxeles circundantes según la distribución gaussiana. Se utiliza para suavizar imágenes de manera más natural y efectiva, reduciendo el ruido sin eliminar información importante.

## FILTRO SOBEL
El operador Sobel es un tipo de filtro que se utiliza para la detección de bordes en imágenes. Se define por un tamaño de matriz o ventana específico, como 3x3 o 5x5. Realiza una medición de gradiente espacial 2D en la imagen, enfatizando así las regiones de alta frecuencia espacial correspondientes a los bordes. Se utiliza habitualmente para determinar la magnitud absoluta aproximada del gradiente en cada punto de una imagen de entrada en escala de grises. Este filtro aplica dos convoluciones en paralelo: una en la dirección horizontal (Gx) y otra en la dirección vertical (Gy). Este operador combina suavizado y diferenciación gaussiana.

## FILTRO DE SUAVIZACIÓN
Para poder aplicar un filtro de suavización sobre una imagen se utiliza un kernel de suavización, usualmente una matriz 3x3, con valores diseñados para reducir el ruido y suavizar los píxeles de una imagen. Para implementarlo, se realiza una convolución entre el kernel de suavización y la imagen dada, multiplicando los valores del kernel por los valores de los píxeles de la imagen, sumando los resultados para representar cada píxel correctamente. Como resultado final, se obtiene la imagen derivada de la original con el filtro de suavización aplicado.

## FILTRO DE NITIDEZ
El filtro de nitidez es una técnica esencial en el procesamiento digital de imágenes para realzar detalles y contornos mediante convoluciones. Resalta las transiciones abruptas en la intensidad luminosa, enfatizando bordes y mejorando el contraste local. Se logra mediante dos métodos principales: la máscara laplaciana y el unsharp masking. La máscara laplaciana detecta y acentúa los cambios bruscos de intensidad, mientras que el unsharp masking suaviza la imagen y luego resta la versión suavizada de la original para aislar y potenciar detalles. Estas técnicas transforman localmente los valores de los píxeles para mejorar la definición sin comprometer la calidad global.

## FILTRO LAPLACIANO
El filtro Laplaciano es una matriz de coeficientes utilizada para modificar cada píxel de la imagen en función de sus píxeles vecinos, detectando y enfatizando bordes. Se aplica para reducir la sensibilidad al ruido en imágenes previamente suavizadas con filtros como el Gaussiano. Este filtro amplifica los componentes de alta frecuencia afinando los bordes y suprimiendo el ruido. Variantes como el Laplacian del Gaussiano (LoG) combinan suavizado y detección de bordes para obtener mejores resultados en el análisis de imágenes.

## PROPUESTA
El filtro propuesto se basa en la integración secuencial de tres efectos principales: aumento de ruido, aplicación del filtro Sobel y del filtro Laplaciano, complementados con un padding circular “wrap around” para asegurar la continuidad en los bordes durante la convolución.

Se introduce variación aleatoria en la intensidad de los píxeles para generar textura, luego se aplica el filtro Sobel para resaltar bordes y, finalmente, el filtro Laplaciano para enfatizar cambios de intensidad, creando una estética impactante y coherente.

## REFERENCIAS
- Esri. (s.f.) Función de convolución. https://doc.arcgis.com/es/allsource/1.0/analysis/raster-functions/convolution-function.htm
- Fisher, R., Perkins, S., & Walker, A. (2003a). Gaussian Smoothing. Hypermedia Image Processing Reference. https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
- Fisher, R., Perkins, S., & Walker, A. (2003b). Sobel Edge Detector. Hypermedia Image Processing Reference. https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
- Función de convolución—ArcMap. (s.f.). Documentación. https://desktop.arcgis.com/es/arcmap/latest/manage-data/raster-and-images/convolution-function.htm
- OpenCV. (2025) Smoothing Images. https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- Prajapati, P. (2024, marzo 5). ¿Cómo se puede utilizar un filtro laplaciano para afinar los bordes de una imagen? LinkedIn. https://es.linkedin.com/advice/3/how-can-you-use-laplacian-filter-sharpen-rjxqe?lang=es&lang=es
- PTC. (2024). Suavizado. https://support.ptc.com/help/mathcad/r10.0/es/index.html#page/PTC_Mathcad_Help/smoothing.html


