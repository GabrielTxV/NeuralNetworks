import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Verifica se há GPUs disponíveis
print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))

# Carrega o dataset Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

# Verifica um índice de dados válido
data_idx = 0  # Exemplo de índice válido
print(valid_images[data_idx])

# Define o número de classes
number_of_classes = train_labels.max() + 1

# Cria o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')  # Adiciona função de ativação
])

# Compila o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Treina o modelo
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)

# Faz previsões
predictions = model.predict(train_images[0:10])

data_idx = 7  # Número do dataset até 59999.

# Exibe a imagem
plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

# Exibe as probabilidades previstas
x_values = range(number_of_classes)
plt.figure()
plt.bar(x_values, model.predict(train_images[data_idx:data_idx+1]).flatten())
plt.xticks(range(10))
plt.show()

# Dicionário de classes
class_names = {
    0: "Camiseta/top",
    1: "Calça",
    2: "Suéter",
    3: "Vestido",
    4: "Casaco",
    5: "Sandália",
    6: "Camisa",
    7: "Tênis",
    8: "Bolsa",
    9: "Bota"
}

# Imprime a classe correta e a classe prevista
predicted_class = np.argmax(predictions[data_idx])
correct_class = train_labels[data_idx]

print("Resposta correta:", class_names[correct_class])
print("Classe prevista:", class_names[predicted_class])

# Verifica se a previsão está correta
if predicted_class == correct_class:
    print("A previsão do modelo está correta.")
else:
    print("A previsão do modelo está incorreta.")