import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gzip


# import torch.optim as optim


filenames = {
    tot: {
        name: f"../mnist/train-{name}-idx{n}-ubyte.gz"
        for name, n in [("images", 3), ("labels", 1)]
    }
    for tot, prefix in [("train", "train"), ("test", "t10k")]
}

image_size = 28
num_images = 60000


def data_reader(tot, iol, hn):
    assert iol in ["images", "labels"]

    with gzip.open(filenames[tot][iol], "r") as f:
        f.read(hn)  # Header
        buf = f.read(image_size * image_size * num_images)
        if iol == "images":
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, image_size, image_size)
        elif iol == "labels":
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.int)
            data = data.reshape(num_images)
        else:
            assert Exception("Unreachable")

    data = torch.tensor(data)
    return data


train_images = data_reader("train", "images", 16)


train_labels = data_reader("train", "labels", 8)


print("Images shape")
print(train_images.shape)
print()

print("Labels shape")
print(train_labels.shape)
print()

# plt.imshow(train_images[0])
# plt.show()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(784, 100)
        self.l2 = nn.Linear(100, 10)
        # self.l1 = nn.Linear(2, 2)
        # self.l2 = nn.Linear(2, 2)

    def forward(self, *xs):
        out = []
        for x in xs:
            x = x.view(x.size(0), -1)
            x = self.l1(x)
            x = nn.functional.relu(x)
            x = self.l2(x)

            out.append(x)

        if len(out) == 1:
            return out[0]

        return out


class Optimizer:
    def __init__(self, params, needs_loss_function):
        self.params = params
        self.needs_loss_function = needs_loss_function

    def zero_grad(self):
        for p in self.params():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class SGD0(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, needs_loss_function=False)
        self.lr = lr

    def step(self):

        for p in self.params():

            g = p.grad
            p.data = p.data - self.lr * g


class SGD1(Optimizer):
    def __init__(self, params, lr0, lrlr):
        super().__init__(params, needs_loss_function=True)
        self.lr = lr0
        self.lrlr = lrlr
        self.lr_store = [lr0]

    def step(self, loss_fn, loss_fn_args):

        initial_parameters_and_grads = [(p.data.clone(), p.grad) for p in self.params()]

        losses = []

        lrs = [self.lr * i for i in [1, self.lrlr]]
        for lr in lrs:
            for p, (p_init, g) in zip(self.params(), initial_parameters_and_grads):
                p.data = p_init - lr * g

            losses.append(loss_fn(**loss_fn_args))

        ix_min = min(range(len(losses)), key=lambda i: losses[i])
        if ix_min == 0:
            self.lr /= self.lrlr
        elif ix_min == 1:
            self.lr *= self.lrlr
        else:
            raise Exception("Unreachable")

        self.lr_store.append(self.lr)

        for p, (p_init, g) in zip(self.params(), initial_parameters_and_grads):
            p.data = p_init.data - self.lr * g


class Experiment:
    def __init__(self, optimiser, n_epochs, optimiser_args=None):
        if optimiser_args is None:
            optimiser_args = {}

        self.n_epochs = n_epochs

        self.nn = MLP()
        self.optimiser = optimiser(self.nn.parameters, **optimiser_args)
        self.loss = nn.functional.cross_entropy
        self.batch_size = 1024

    def stats(self, images, labels):

        with torch.no_grad():
            y_hat_train = self.nn.forward(images)
            loss_train = self.loss(y_hat_train, labels).item()
            n_correct = (labels == y_hat_train.argmax(dim=1)).sum()
            n_correct = float(n_correct.numpy())
            accuracy = n_correct / len(train_labels)

            stats = {"accuracy": accuracy, "loss": loss_train}
            return stats

    def loss_fn(self, images, labels):
        return self.stats(images, labels)["loss"]

    def train_one_epoch(self, train_x, train_y):
        x_batches = torch.split(train_x, self.batch_size)
        y_batches = torch.split(train_y, self.batch_size)

        for x_batch, y_batch in zip(x_batches, y_batches):
            self.optimiser.zero_grad()
            y_hat_batch = self.nn.forward(x_batch)
            loss_batch = self.loss(y_hat_batch, y_batch)
            loss_batch.backward()

            if self.optimiser.needs_loss_function:

                self.optimiser.step(
                    self.loss_fn, {"images": x_batch, "labels": y_batch}
                )
            else:
                self.optimiser.step()

    def run_and_return_stats(self, verbose=False):

        if verbose:
            print("Parameters shape")
            for i in list(self.nn.parameters()):
                print(i.shape)
                print()

        for epoch in range(self.n_epochs):
            if verbose:
                print(f"Epoch: {epoch}")
            self.train_one_epoch(train_images, train_labels)

            stats = self.stats(train_images, train_labels)

            if verbose:
                loss_train, accuracy = [stats[x] for x in ["loss", "accuracy"]]
                print(f"Loss: {loss_train}")
                print(f"Accuracy: {100 * accuracy : .1f}%")
                print()

        if self.optimiser.needs_loss_function:
            plt.plot(self.optimiser.lr_store)
            plt.show()

        return {"train_" + i: stats[i] for i in stats}


def main(test=True):

    n_epochs = 8
    # SDG0

    if test:
        lrs = [0.005]
    else:
        lrs = [10 ** (-1 * i) for i in np.linspace(2, 5, num=8)]

    lrs = sorted(lrs)

    losses = []
    acs = []

    for lr in lrs:

        e = Experiment(SGD0, n_epochs, {"lr": lr})
        run_stats = e.run_and_return_stats(verbose=False)
        loss, ac = [run_stats["train_" + x] for x in ["loss", "accuracy"]]
        losses.append(loss)
        acs.append(ac)
        print(f"{lr:8.6f} {loss:8.3f} {ac:8.3f}")

    # SGD1

    e = Experiment(SGD1, n_epochs // 2, {"lr0": 0.001, "lrlr": 1.03})
    run_stats = e.run_and_return_stats(verbose=False)

    loss, ac = [run_stats["train_" + x] for x in ["loss", "accuracy"]]
    print("loss     acc")
    print(f"{loss:8.3f} {ac:8.3f}")

    plt.show()


main(test=True)
