# Importando as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Carregando o dataset MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Pré-processando os dados
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Convertendo as labels para categorias one-hot
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Construindo o modelo
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

# Compilando o modelo
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinando o modelo
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Selecionar uma imagem do dataset
index = 59999   # Escolher índice da imagem de 0 a 59999
selected_image = train_images[index].reshape(28, 28)
selected_label = train_labels[index].argmax()

# Exibir a imagem selecionada
plt.imshow(selected_image, cmap='gray')
plt.title(f"Label: {selected_label}")
plt.axis('off')
plt.show()

# Fazer a previsão para a imagem selecionada
selected_image = selected_image.reshape(1, 28 * 28)  # Redimensionar para o formato esperado pelo modelo
prediction = model.predict(selected_image)
predicted_label = prediction.argmax()

print(f"Rótulo real: {selected_label}")
print(f"Rótulo previsto: {predicted_label}")

# Avaliando o modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Acurácia no conjunto de teste: {test_acc}")