# Importando as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregando o dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalizando os dados (os valores dos pixels estarão entre 0 e 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Convertendo as labels para categorias one-hot
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Construindo o modelo CNN
model = models.Sequential()

# Camada convolucional 1
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Camada convolucional 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Camada convolucional 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Camada Flatten
model.add(layers.Flatten())

# Camada totalmente conectada (fully connected)
model.add(layers.Dense(64, activation='relu'))

# Camada de saída: 10 neurônios (10 classes do CIFAR-10), com ativação softmax
model.add(layers.Dense(10, activation='softmax'))

# Compilando o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Avaliando o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Acurácia no conjunto de teste: {test_acc}")

# -------- Previsão em Imagens Reais -------- #
# Carregar uma imagem real para teste
img_path = 'hulk.jpg'  # Substitua pelo caminho da imagem escolhida

# Carregar e redimensionar a imagem para 32x32 pixels, que é o formato esperado pelo modelo
img = image.load_img(img_path, target_size=(32, 32))

# Converter a imagem para um array de NumPy e normalizar os valores dos pixels (0 a 1)
img_array = image.img_to_array(img) / 255
img_array = np.expand_dims(img_array, axis=0)  # Adicionar a dimensão do lote (batch)

# Fazer a previsão
predicao = model.predict(img_array)

# Mostrar a classe prevista (a classe com a maior probabilidade)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classe_prevista = classes[np.argmax(predicao)]
print(f"Classe prevista: {classe_prevista}")
