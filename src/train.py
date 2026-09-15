## train the classifier and save the best model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras

import config
import data
from model import make_model

def main():
    train_ds = data.make_train_ds(augmentations=data.AUGMENTATION)
    val_ds = data.make_evaluation_ds(config.VAL_DIR)
    class_names = sorted(train_ds.class_indices, key=train_ds.class_indices.get)
    print('classes:', class_names)

    m = make_model(num_classes=len(class_names))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        str(config.MODELS_DIR/"xception_{epoch:02d}-{val_accuracy:.2f}.keras"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    )

    history = m.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint],
    )

    hist = history.history
    plt.plot(hist["accuracy"], label="train_accuracy")
    plt.plot(hist["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.PLOTS_DIR/"history.png", dpi=120,bbox_inches="tight")
    plt.close()

    print("the best validation accuracy was", max(hist["val_accuracy"]))


if __name__ == "__main__":
    main()