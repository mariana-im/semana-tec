"""
Mariana Islas Mondragón / A01253435
Dentro de este codigo se aplica un kernel de sobel a una imagen utilizando convolución.

Para poder ejecutar este codigo es necesario descargar las librerias numpy, cv2 y matplotlib. También es necesario revisar la dirección
de la imagen cat que se utiliza dentro del codigo, si no es correcta actualizar dirección correcta en linea 45. Después de esto es posible 
utilizar el programa dandole click a ejecutar. Se desplegara una ventana con la imagen original y la imagen con el filtro aplicado.
"""

import numpy as np                # Librería para operaciones numéricas y matrices
import cv2                       # Librería OpenCV para procesamiento de imágenes
import matplotlib.pyplot as plt  # Librería para mostrar imágenes gráficamente

def my_convolution(image_gray, kernel):

    # Obtener dimensiones de la imagen y del filtro
    img_height, img_width = image_gray.shape
    kernel_height, kernel_width = kernel.shape

    # Calcular dimensiones de la imagen de salida (sin padding)
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    # Inicializar matriz de salida con ceros (en float para precisión)
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Realizar la convolución: recorrer cada posición válida de la imagen
    for i in range(output_height):
        for j in range(output_width):
            # Extraer la región de la imagen del mismo tamaño que el filtro
            region = image_gray[i:i+kernel_height, j:j+kernel_width]
            # Multiplicar elemento a elemento (region * kernel) y sumar los valores
            value = np.sum(region * kernel)
            # Guardar el resultado en la matriz de salida
            output[i, j] = value

    # Asegurar que los valores estén entre 0 y 255 (rango de imagen)
    output = np.clip(output, 0, 255)

    # Retornar la imagen convertida a tipo uint8 (valores enteros de 0-255)
    return output.astype(np.uint8)

if __name__ == '__main__':
    # Especificar la ruta de la imagen a procesar
    image_path = r'Actividades/cat.jpg'  # Puedes cambiar este nombre por cualquier imagen que tengas

    # Leer la imagen en formato BGR (por defecto en OpenCV)
    image_bgr = cv2.imread(image_path)

    # Verificar que la imagen se haya cargado correctamente
    if image_bgr is None:
        print("Error: No se pudo cargar la imagen.")
        exit()  # Salir del programa si la imagen no existe

    # Convertir imagen de BGR (OpenCV) a RGB (para matplotlib)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convertir la imagen a escala de grises para aplicar convolución
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Definir un filtro Sobel para detectar bordes horizontales
    sobel_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Aplicar la convolución a la imagen en escala de grises
    resultado = my_convolution(image_gray, sobel_kernel)

    # Mostrar la imagen original (color) y el resultado (gris) lado a lado
    plt.figure(figsize=(12, 6))  # Tamaño de la figura (ancho x alto en pulgadas)

    # Mostrar imagen original en color
    plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, posición 1
    plt.imshow(image_rgb)  # Mostrar imagen en RGB
    plt.title('Imagen Original (Color)')  # Título del subplot
    plt.axis('off')  # Ocultar ejes

    # Mostrar imagen resultante de la convolución
    plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, posición 2
    plt.imshow(resultado, cmap='gray')  # Mostrar en escala de grises
    plt.title('Resultado de Convolución (Sobel)')  # Título del subplot
    plt.axis('off')  # Ocultar ejes

    plt.tight_layout()  # Ajustar espacio entre las imágenes
    plt.show()  # Mostrar la ventana con ambas imágenes

    # Guardar el resultado como archivo de imagen
    cv2.imwrite('resultado_convolucion.jpg', resultado)
    print("Imagen resultante guardada como 'resultado_convolucion.jpg'")
