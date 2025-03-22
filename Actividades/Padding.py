"""
Mariana Islas Mondragón / A01253435
Dentro de este codigo se aplica un kernel de sobel a una imagen utilizando convolución y se le agrega padding.

Para poder ejecutar este codigo es necesario descargar las librerías numpy, cv2 y matplotlib. También es necesario revisar la dirección
de la imagen cat que se utiliza dentro del codigo, si no es correcta actualizar dirección correcta en linea 46. Después de esto es posible 
utilizar el programa dandole click a ejecutar. Se desplegara una ventana con la imagen original, la imagen original con padding
y la imagen con el filtro aplicado y padding.
"""
# Importación de diferentes librerías
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Función para realizar convolución, aplicar kernel y padding a imagen original.
def my_convolution(image_gray, kernel, use_padding=True):
    # Se definen las dimensiones de la imagen y el kernel para poder realizar la convolución
    img_height, img_width = image_gray.shape
    kernel_height, kernel_width = kernel.shape

    # Calcular padding necesario (mitad del tamaño del kernel)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Si use_padding es true se realizara el siguiente codigo, en donde se agrega padding alrededor de la imagen
    if use_padding:
        # Imagen con padding de ceros alrededor
        padded_image = np.pad(image_gray, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        output_height, output_width = img_height, img_width  # Mantiene tamaño original
    # Si use_padding es falso se regresara la imagen sin padding
    else:
        padded_image = image_gray  # Sin padding
        output_height = img_height - kernel_height + 1
        output_width = img_width - kernel_width + 1

    # Inicializar imagen de salida
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Aplicar convolución recorriendo la imagen con padding
    for i in range(output_height):
        for j in range(output_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return padded_image, np.clip(output, 0, 255).astype(np.uint8) # Se regresa la imagen con padding y el resultado de la convolución

if __name__ == '__main__':
    image_path = r'Actividades\cat.jpg'  # Se almacena la imagen
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Se almacena la imagen en escala de grises

    if image_gray is None:
        print("Error: No se pudo cargar la imagen.")
        exit()

    # Kernel de filtro sobel
    sobel_kernel = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

    # Guardar imagen con padding y filtro y resultado de convolución
    padded_image, resultado_con_padding = my_convolution(image_gray, sobel_kernel, use_padding=True)

    # Mostrar imágenes
    plt.figure(figsize=(18, 6))

    # Mostrar imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    # Mostrar imagen con padding (antes de la convolución)
    plt.subplot(1, 3, 2)
    plt.imshow(padded_image, cmap='gray')
    plt.title('Imagen con Padding (Antes de la Convolución)')
    plt.axis('off')

    # Mostrar imagen resultante de la convolución con padding
    plt.subplot(1, 3, 3)
    plt.imshow(resultado_con_padding, cmap='gray')
    plt.title('Resultado Convolución usando Filtro Sobel y con Padding')
    plt.axis('off')

    plt.tight_layout() #Ajustar layout de subplots
    plt.show() # Mostrar resultado