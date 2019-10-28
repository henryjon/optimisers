import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# import torch.optim as optim

import gzip

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


filenames = {
    "train": {
        "images": "../mnist/train-images-idx3-ubyte.gz",
        "labels": "../mnist/train-labels-idx1-ubyte.gz",
    }
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
        self.hidden_size = 20
        self.l1 = nn.Linear(784, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, 10)

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


class Optimizer_smoothed(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_size_ewma = None
        self.smoothing = 0.1

    def grad_norm(self):

        return sum(p.grad.norm() for p in self.params())

    def update_grad_size_ewma(self):
        grad_norm = self.grad_norm()

        if self.grad_size_ewma is None:
            self.grad_size_ewma = grad_norm

        self.grad_size_ewma = (self.smoothing) * grad_norm + (
            1 - self.smoothing
        ) * self.grad_size_ewma


class SGD0(Optimizer_smoothed):
    def __init__(self, lr, **kwargs):
        super().__init__(needs_loss_function=False, **kwargs)
        self.lr = lr

    def stats(self):
        return {"lr": self.lr}

    def step(self):

        self.update_grad_size_ewma()

        grad_norm = self.grad_norm()
        for p in self.params():

            g = p.grad
            p.data = p.data - self.lr * self.grad_size_ewma * (g / grad_norm)


class SGD1(Optimizer_smoothed):
    def __init__(self, lr0, lrlr, update_freq=1, **kwargs):
        super().__init__(needs_loss_function=True, **kwargs)
        self.lr0 = lr0
        self.lr = lr0
        self.lrlr = lrlr
        self.lrs = []
        self.update_freq = update_freq
        self.count = 0
        self.losses = []
        self.grad_size_ewma = None

    def stats(self):
        return {"lr0": self.lr0, "lr": self.lr, "lrlr": self.lrlr}

    def step(self, loss_fn, loss_fn_args):

        initial_parameters_and_grads = [(p.data.clone(), p.grad) for p in self.params()]

        if (self.count % self.update_freq) == 0:

            losses = []

            lrs = [self.lr * i for i in [1, self.lrlr]]
            for lr in lrs:
                for p, (p_init, g) in zip(self.params(), initial_parameters_and_grads):
                    p.data = p_init - lr * g

                losses.append(loss_fn(**loss_fn_args))

            loss = min(losses)
            ix_min = min(range(len(losses)), key=lambda i: losses[i])

            if ix_min == 0:
                # Reduce the learning rate before updating
                self.lr /= self.lrlr
            elif ix_min == 1:
                # Increase the learning rate before updating
                self.lr *= self.lrlr
            else:
                raise Exception("Unreachable")
        else:
            loss = loss_fn(**loss_fn_args)

        self.update_grad_size_ewma()

        grad_norm = self.grad_norm()
        for p, (p_init, g) in zip(self.params(), initial_parameters_and_grads):
            p.data = p_init.data - self.lr * self.grad_size_ewma * (g / grad_norm)

        self.count += 1
        self.lrs.append(self.lr)
        self.losses.append(loss)


class Experiment:
    def __init__(self, optimizer, n_epochs, optimizer_args=None):
        if optimizer_args is None:
            optimizer_args = {}

        self.n_epochs = n_epochs

        self.nn = MLP()

        self.optimizer = optimizer(params=self.nn.parameters, **optimizer_args)
        self.loss = nn.functional.cross_entropy
        self.batch_size = 1000
        self.start_time = time.time()
        self.stats = None

    def stats_on(self, images, labels):

        with torch.no_grad():
            y_hat_train = self.nn.forward(images)
            loss_train = self.loss(y_hat_train, labels).item()
            n_correct = (labels == y_hat_train.argmax(dim=1)).sum()
            n_correct = float(n_correct.numpy())
            accuracy = n_correct / len(train_labels)

            stats = {"accuracy": accuracy, "loss": loss_train}
            stats.update(self.optimizer.stats())
            return stats

    def loss_fn(self, images, labels):
        return self.stats_on(images, labels)["loss"]

    def train_one_epoch(self, train_x, train_y):
        x_batches = torch.split(train_x, self.batch_size)
        y_batches = torch.split(train_y, self.batch_size)

        for x_batch, y_batch in zip(x_batches, y_batches):
            self.optimizer.zero_grad()
            y_hat_batch = self.nn.forward(x_batch)
            loss_batch = self.loss(y_hat_batch, y_batch)
            loss_batch.backward()

            if self.optimizer.needs_loss_function:

                self.optimizer.step(
                    self.loss_fn, {"images": x_batch, "labels": y_batch}
                )
            else:
                self.optimizer.step()

    def run(self, verbose=False, plot_lrs=False):

        if verbose:
            print("Parameters shape")
            for i in list(self.nn.parameters()):
                print(i.shape)
                print()

        for epoch in range(self.n_epochs):
            if verbose:
                print(f"Epoch: {epoch}")
            self.train_one_epoch(train_images, train_labels)

            stats = self.stats_on(train_images, train_labels)

            if verbose:
                loss_train, accuracy = [stats[x] for x in ["loss", "accuracy"]]
                print(f"Loss: {loss_train}")
                print(f"Accuracy: {100 * accuracy : .1f}%")
                print()

        stop_time = time.time()

        stats["run_time"] = stop_time - self.start_time
        self.stats = stats

        if plot_lrs:

            x = np.linspace(0, self.n_epochs, num=self.optimizer.count)

            plt.title("Log learning rate")
            plt.xlabel("epoch")
            plt.plot(x, np.log10(self.optimizer.lrs))
            name = f"lr0-{self.optimizer.lr0}-hidden_size-{self.nn.hidden_size}.png"

            plt.savefig("plots/lr-" + name)
            plt.clf()

            plt.title("Log loss")
            plt.xlabel("epoch")
            plt.plot(x[10:], np.log10(self.optimizer.losses)[10:])

            plt.savefig("plots/loss-" + name)
            plt.clf()


def main(test):

    if test:
        lrs = [0.005, 0.01]
        n_runs = 3
        n_epochs = 5
    else:
        lrs = [10 ** (-1 * i) for i in np.linspace(2, 5, num=10)]
        n_runs = 10
        n_epochs = 10

    lrs = sorted(lrs)
    lr0s = lrs.copy()

    def string_from_list(dic_list, sort_column):
        df = pd.DataFrame(dic_list)
        df = pd.merge(
            df.groupby("id").mean(),
            df.groupby("id").std(),
            on="id",
            suffixes=("_mean", "_std"),
        )

        df.sort_values(sort_column, inplace=True)
        return df.to_string(
            columns=[
                sort_column,
                "accuracy_mean",
                "accuracy_std",
                "loss_mean",
                "loss_std",
            ],
            index=False,
            formatters={
                k: v.format
                for k, v in {
                    "loss_mean": "{:.2f}",
                    "loss_std": "{:.2f}",
                    "accuracy_mean": "{:.1%}",
                    "accuracy_std": "{:.1%}",
                }.items()
            },
        )
        return df

    # SDG0

    stats_dics = []
    for count in range(n_runs):
        print(count)
        for lr in lrs:

            e = Experiment(SGD0, n_epochs, {"lr": lr})
            e.run(verbose=False)
            e.stats["count"] = count
            e.stats["id"] = str(lr)
            stats_dics.append(e.stats)
    summary_str0 = string_from_list(stats_dics, sort_column="lr_mean")
    print(summary_str0)

    # SGD1

    stats_dics = []
    for count in range(n_runs):
        print(count)
        for lr0 in lr0s:
            e = Experiment(SGD1, n_epochs, {"lr0": lr0, "lrlr": 1.01})
            plot_lrs = count == n_runs - 1
            e.run(verbose=False, plot_lrs=plot_lrs)
            e.stats["count"] = count
            e.stats["id"] = str(lr0)
            stats_dics.append(e.stats)

    summary_str1 = string_from_list(stats_dics, sort_column="lr0_mean")
    print(summary_str1)


if __name__ == "__main__":
    main(test=False)
