import glob
import os
from collections import defaultdict
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from keras.callbacks import Callback
import pandas as pd

from kutilities.helpers.data_preparation import onehot_to_categories
from kutilities.helpers.generic import get_model_desc
from kutilities.helpers.ui import move_figure

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'],
                  'monospace': ['Computer Modern Typewriter']})
# plt.rc('text', usetex=True)
plt.rc("figure", facecolor="white")


class LossEarlyStopping(Callback):
    def __init__(self, metric, value, mode="less"):
        super().__init__()
        self.metric = metric
        self.value = value
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        if self.mode == "less":
            if logs[self.metric] < self.value:
                self.model.stop_training = True
                print('Early stopping - {} is {} than {}'.format(self.metric,
                                                                 self.mode,
                                                                 self.value))
        if self.mode == "more":
            if logs[self.metric] > self.value:
                self.model.stop_training = True
                print('Early stopping - {} is {} than {}'.format(self.metric,
                                                                 self.mode,
                                                                 self.value))


class MetricsCallback(Callback):
    """

    """

    def __init__(self, metrics, datasets, regression=False, batch_size=512):
        super().__init__()
        self.datasets = datasets
        self.metrics = metrics
        self.regression = regression
        self.batch_size = batch_size

    @staticmethod
    def predic_classes(predictions):
        if predictions.shape[-1] > 1:
            return predictions.argmax(axis=-1)
        else:
            return (predictions > 0.5).astype('int32')

    def add_predictions(self, dataset, name="set", logs={}):
        X = dataset[0]
        y = dataset[1]

        if self.regression:
            y_pred = self.model.predict(X, batch_size=self.batch_size,
                                        verbose=0)
            y_pred = numpy.reshape(y_pred, y.shape)
            y_test = y

        # test if the labels are categorical or singular
        else:
            if len(y.shape) > 1:
                try:
                    y_pred = self.model.predict_classes(X,
                                                        batch_size=self.batch_size,
                                                        verbose=0)
                except:
                    y_pred = self.predic_classes(
                        self.model.predict(X, batch_size=self.batch_size,
                                           verbose=0))

                y_test = onehot_to_categories(y)

            else:
                y_pred = self.model.predict(X, batch_size=self.batch_size,
                                            verbose=0)
                y_pred = numpy.array([int(_y > 0.5) for _y in y_pred])
                y_test = y

        for k, metric in self.metrics.items():
            score = numpy.squeeze(metric(y_test, y_pred))
            entry = ".".join([name, k])
            self.params['metrics'].append(entry)
            logs[entry] = score

    def on_epoch_end(self, epoch, logs={}):
        for name, data in self.datasets.items():
            self.add_predictions(data if len(data) > 1 else data[0], name=name,
                                 logs=logs)

        data = {}
        for dataset in sorted(self.datasets.keys()):
            data[dataset] = {metric: logs[".".join([dataset, metric])]
                             for metric in sorted(self.metrics.keys())}
        print(pd.DataFrame.from_dict(data, orient="index"))
        print()


class PlottingCallback(Callback):
    def __init__(self, benchmarks=None, grid_ranges=None, width=4, height=4,
                 plot_name=None):
        super().__init__()
        self.height = height
        self.width = width
        self.benchmarks = benchmarks
        self.grid_ranges = grid_ranges
        self.model_loss = []
        self.validation_loss = []
        self.custom_metrics = defaultdict(list)
        self.fig = None

        res_path = os.path.join(os.getcwd(), 'experiments')
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        models = len(glob.glob(os.path.join(res_path, "model*.png")))

        if plot_name is None:
            self.plot_fname = os.path.join(res_path,
                                           'model_{}.png'.format(models + 1))
        else:
            self.plot_fname = os.path.join(res_path,
                                           '{}.png'.format(plot_name))

    def on_train_begin(self, logs={}):
        sns.set_style("whitegrid")
        sns.set_style("whitegrid", {"grid.linewidth": 0.5,
                                    "lines.linewidth": 0.5,
                                    "axes.linewidth": 0.5})
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e",
                  "#2ecc71"]
        sns.set_palette(sns.color_palette(flatui))
        # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        # sns.set_palette(sns.color_palette("Set2", 10))

        plt.ion()  # set plot to animated
        width = self.width * (1 + len(self.get_metrics(logs)))
        height = self.height
        self.fig = plt.figure(figsize=(width, height))

        # move it to the upper left corner
        move_figure(self.fig, 25, 25)

    def save_plot(self):
        self.fig.savefig(self.plot_fname, dpi=100)

    @staticmethod
    def get_metrics(logs):
        custom_metrics_keys = defaultdict(list)
        for entry in logs.keys():
            metric = entry.split(".")
            if len(metric) > 1:  # custom metric
                custom_metrics_keys[metric[0]].append(metric[1])
        return custom_metrics_keys

    def on_epoch_end(self, epoch, logs={}):
        self.fig.clf()
        linewidth = 1.2
        self.fig.set_size_inches(
            self.width * (1 + len(self.get_metrics(logs))),
            self.height, forward=True)
        custom_metrics_keys = self.get_metrics(logs)

        total_plots = len(custom_metrics_keys) + 1
        ##################################################
        # First - Plot Models loss
        self.model_loss.append(logs['loss'])
        self.validation_loss.append(logs['val_loss'])

        ax = self.fig.add_subplot(1, total_plots, 1)
        ax.plot(self.model_loss, linewidth=linewidth)
        ax.plot(self.validation_loss, linewidth=linewidth)
        ax.set_title('model loss', fontsize=10)
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train', 'val'], loc='upper left', fancybox=True)
        ax.grid(True)
        ax.grid(b=True, which='major', color='gray', linewidth=.5)
        ax.grid(b=True, which='minor', color='gray', linewidth=0.5)
        ax.tick_params(labelsize=10)
        # leg = ax.gca().get_legend()

        ##################################################
        # Second - Plot Custom Metrics
        for i, (dataset_name, metrics) in enumerate(
                sorted(custom_metrics_keys.items(), reverse=False)):
            axs = self.fig.add_subplot(1, total_plots, i + 2)
            axs.set_title(dataset_name, fontsize=10)
            axs.set_ylabel('score')
            axs.set_xlabel('epoch')
            if self.grid_ranges:
                axs.set_ylim(self.grid_ranges)

            # append the values to the corresponding array
            for m in sorted(metrics):
                entry = ".".join([dataset_name, m])
                self.custom_metrics[entry].append(logs[entry])
                axs.plot(self.custom_metrics[entry], label=m,
                         linewidth=linewidth)

            axs.tick_params(labelsize=10)
            labels = list(sorted(metrics))
            if self.benchmarks:
                for (label, benchmark), color in zip(self.benchmarks.items(),
                                                     ["y", "r"]):
                    axs.axhline(y=benchmark, linewidth=linewidth, color=color)
                    labels = labels + [label]
            axs.legend(labels, loc='upper left', fancybox=True)
            axs.grid(True)
            axs.grid(b=True, which='major', color='gray', linewidth=.5)
            axs.grid(b=True, which='minor', color='gray', linewidth=0.5)

        plt.rcParams.update({'font.size': 10})

        desc = get_model_desc(self.model)
        self.fig.text(.02, .02, desc, verticalalignment='bottom', wrap=True,
                      fontsize=8)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=.18)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.save_plot()

    def on_train_end(self, logs={}):
        plt.close(self.fig)
        # self.save_plot()


class WeightsCallback(Callback):
    # todo: update to Keras 2.X
    def __init__(self, parameters=None, stats=None, merge_weights=True):
        super().__init__()
        self.layers_stats = defaultdict(dict)
        self.fig = None
        self.parameters = parameters
        self.stats = stats
        self.merge_weights = merge_weights
        if parameters is None:
            self.parameters = ["W"]
        if stats is None:
            self.stats = ["mean", "std"]

    def get_trainable_layers(self):
        layers = []
        for layer in self.model.layers:
            if "merge" in layer.name:
                for l in layer.layers:
                    if hasattr(l, 'trainable') and l.trainable and len(
                            l.weights):
                        if not any(x.name == l.name for x in layers):
                            layers.append(l)
            else:
                if hasattr(layer, 'trainable') and layer.trainable and len(
                        layer.weights):
                    layers.append(layer)
        return layers

    def on_train_begin(self, logs={}):
        for layer in self.get_trainable_layers():
            for param in self.parameters:
                if any(w for w in layer.weights if param in w.name.split("_")):
                    name = layer.name + "_" + param
                    self.layers_stats[name]["values"] = numpy.asarray(
                        []).ravel()
                    for s in self.stats:
                        self.layers_stats[name][s] = []

        # plt.style.use('ggplot')
        plt.ion()  # set plot to animated
        width = 3 * (1 + len(self.stats))
        height = 2 * len(self.layers_stats)
        self.fig = plt.figure(figsize=(width, height))
        # sns.set_style("whitegrid")
        self.draw_plot()

    def draw_plot(self):
        self.fig.clf()

        layers = self.get_trainable_layers()
        height = len(self.layers_stats)
        width = len(self.stats) + 1

        plot_count = 1
        for layer in layers:
            for param in self.parameters:
                weights = [w for w in layer.weights if
                           param in w.name.split("_")]

                if len(weights) == 0:
                    continue

                val = numpy.column_stack((w.get_value() for w in weights))
                name = layer.name + "_" + param

                self.layers_stats[name]["values"] = val.ravel()
                ax = self.fig.add_subplot(height, width, plot_count)
                ax.hist(self.layers_stats[name]["values"], bins=50)
                ax.set_title(name, fontsize=10)
                ax.grid(True)
                ax.tick_params(labelsize=8)
                plot_count += 1

                for s in self.stats:
                    axs = self.fig.add_subplot(height, width, plot_count)

                    if s == "raster":
                        if len(val.shape) > 2:
                            val = val.reshape((val.shape[0], -1), order='F')
                        self.layers_stats[name][s] = val
                        m = axs.imshow(self.layers_stats[name][s],
                                       cmap='coolwarm',
                                       interpolation='nearest',
                                       aspect='auto', )  # aspect='equal'
                        cbar = self.fig.colorbar(mappable=m)
                        cbar.ax.tick_params(labelsize=8)
                    else:
                        self.layers_stats[name][s].append(
                            getattr(numpy, s)(val))
                        axs.plot(self.layers_stats[name][s])
                        axs.set_ylabel(s, fontsize="small")
                        axs.set_xlabel('epoch', fontsize="small")
                        axs.grid(True)

                    axs.set_title(name + " - " + s, fontsize=10)
                    axs.tick_params(labelsize=8)
                    plot_count += 1

        # plt.figtext(.1, .1, get_model_desc(self.model), wrap=True, fontsize=8)
        desc = get_model_desc(self.model)
        self.fig.text(.02, .02, desc, verticalalignment='bottom', wrap=True,
                      fontsize=8)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=.14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self):

        layers = self.get_trainable_layers()

        for layer in layers:
            for param in self.parameters:
                weights = [w for w in layer.weights if
                           param in w.name.split("_")]

                if len(weights) == 0:
                    continue

                val = numpy.column_stack((w.get_value() for w in weights))
                name = layer.name + "_" + param
                self.layers_stats[name]["values"] = val.ravel()
                for s in self.stats:
                    if s == "raster":
                        if len(val.shape) > 2:
                            val = val.reshape((val.shape[0], -1), order='F')
                        self.layers_stats[name][s] = val
                        # self.fig.colorbar()
                    else:
                        self.layers_stats[name][s].append(
                            getattr(numpy, s)(val))

        plt.figtext(.02, .02, get_model_desc(self.model), wrap=True,
                    fontsize=8)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=.2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_epoch_end(self, epoch, logs={}):
        self.update_plot()
