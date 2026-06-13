import matplotlib.pyplot as plt

# Epochs da 81 a 100
epochs = list(range(81, 101))

# Simulazione di crescita dell'accuracy nel tempo
accuracy = [0.92, 0.93, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.973,
            0.975, 0.977, 0.979, 0.981, 0.982, 0.983, 0.983, 0.983, 0.983, 0.983]

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy, marker='o', color='green', label='Accuracy Classificazione')
plt.axhline(0.9833, linestyle='--', color='gray', linewidth=1)
plt.text(81.5, 0.984, 'Accuracy finale: 98.33%', color='green')
plt.title('Andamento dell\'Accuracy - FieldNet')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0.91, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("FieldNet_Accuracy_Graph.png")
plt.show()
