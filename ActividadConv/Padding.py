"""
Mariana Islas Mondragón / A01253435
Dentro de este codigo se aplica un kernel de sobel a una imagen utilizando convolución.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_convolution(image_gray, kernel, use_padding=True):
    """Aplica una convolución 2D, con o sin padding."""

    img_height, img_width = image_gray.shape
    kernel_height, kernel_width = kernel.shape

    # Calcular padding necesario (mitad del tamaño del kernel)
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    if use_padding:
        # Imagen con padding de ceros alrededor
        padded_image = np.pad(image_gray, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        output_height, output_width = img_height, img_width  # Mantiene tamaño original
    else:
        padded_image = image_gray  # Sin padding
        output_height = img_height - kernel_height + 1
        output_width = img_width - kernel_width + 1

    # Inicializar imagen de salida
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # Aplicar convolución recorriendo la imagen
    for i in range(output_height):
        for j in range(output_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)

    return padded_image, np.clip(output, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    image_path = r'C:\Users\maria\Desktop\Actividad\semana-tec\ActividadConv\cat.jpg'  
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        print("Error: No se pudo cargar la imagen.")
        exit()

    # Filtro Sobel para detectar bordes
    sobel_kernel = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

    # Aplicar convolución con padding
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
    plt.title('Resultado Convolución con Padding')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Guardar imágenes
    cv2.imwrite('resultado_con_padding.jpg', resultado_con_padding)
    print("Imagen resultante guardada como 'resultado_con_padding.jpg'")
