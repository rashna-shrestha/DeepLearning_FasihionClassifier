"""Score a saved model on the test split."""

import sys

from tensorflow import keras

import config
import data


def main(model_path):
    m = keras.models.load_model(model_path)
    input_size = m.input_shape[1]

    test_ds = data.make_evaluation_ds(config.TEST_DIR, input_size=input_size)
    loss, accuracy = m.evaluate(test_ds)

    print("test loss:", round(loss, 4))
    print("test accuracy:", round(accuracy, 4))


if __name__ == "__main__":
    main(sys.argv[1])

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         raise SystemExit("usage: python evaluate.py <model_path>")
#     main(sys.argv[1])
