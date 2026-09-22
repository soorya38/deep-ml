import numpy as np

def dummy_classifier(y_train, n_test, strategy, constant=None):

    match strategy:

        case "most_frequent":
            values, counts = np.unique(y_train, return_counts=True)

            # np.unique sorts values, so argmax naturally
            # breaks ties using the smallest label.
            label = values[np.argmax(counts)]

            return [label] * n_test

        case "constant":
            return [constant] * n_test

        case "uniform":
            classes = np.unique(y_train)

            return [
                classes[i % len(classes)]
                for i in range(n_test)
            ]

        case "stratified":
            classes, counts = np.unique(y_train, return_counts=True)

            frequencies = counts / len(y_train)
            expected = n_test * frequencies

            predictions_count = np.floor(expected).astype(int)

            remaining = n_test - predictions_count.sum()

            fractional = expected - predictions_count

            # Largest fractional part first.
            # np.unique gives sorted classes, so ties
            # automatically favor the smallest class.
            order = np.argsort(-fractional)

            for i in order[:remaining]:
                predictions_count[i] += 1

            predictions = []

            for i in range(len(classes)):
                predictions.extend(
                    [classes[i]] * predictions_count[i]
                )

            return predictions

        case _:
            raise ValueError("Unknown strategy")