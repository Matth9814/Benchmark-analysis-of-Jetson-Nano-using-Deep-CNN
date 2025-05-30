import json
import matplotlib.pyplot as plt

plot_list = ["./training_results_fashion_mnist.json","./training_results_cifar10_32b_15epochs.json"]
for file in plot_list:
    fp = open(file)
    res = json.load(fp)
    fp.close()

    # Visualize results
    acc = res['acc']
    val_acc = res['val_acc']

    loss = res['loss']
    val_loss = res['val_loss']

    epochs_range = range(res['epochs'])

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
plt.show()
