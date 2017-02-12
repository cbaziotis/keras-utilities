"""
Created by Christos Baziotis.
"""
import numpy
from keras import backend as K
from keras.callbacks import Callback
import glob
import os
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import matplotlib.transforms as mtrans
from helpers.data_preparation import onehot_to_categories


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
                print('Early stopping - {} is {} than {}'.format(self.metric, self.mode, self.value))
        if self.mode == "more":
            if logs[self.metric] > self.value:
                self.model.stop_training = True
                print('Early stopping - {} is {} than {}'.format(self.metric, self.mode, self.value))


class MetricsCallback(Callback):
    def __init__(self, metrics, validation_data, training_data=None, test_data=None, regression=False):
        super().__init__()
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.metrics = metrics
        self.regression = regression

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
            y_pred = self.model.predict(X, batch_size=2048, verbose=0)
            y_pred = numpy.reshape(y_pred, y.shape)
            y_test = y

        # test if the labels are categorical or singular
        else:
            if len(y.shape) > 1:
                try:
                    y_pred = self.model.predict_classes(X, batch_size=2048, verbose=0)
                except Exception as e:
                    y_pred = self.predic_classes(self.model.predict(X, batch_size=2048, verbose=0))

                y_test = onehot_to_categories(y)

            else:
                y_pred = self.model.predict(X, batch_size=2048, verbose=0)
                y_pred = numpy.array([int(_y > 0.5) for _y in y_pred])
                y_test = y

        for k, metric in self.metrics.items():
            score = numpy.squeeze(metric(y_test, y_pred))
            entry = ".".join([name, k])
            self.params['metrics'].append(entry)
            logs[entry] = score

    @staticmethod
    def get_activations(model, layer, X_batch):
        get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
        activations = get_activations([X_batch, 0])
        return activations

    #
    @staticmethod
    def get_input_mask(model, layer, X_batch):
        get_input_mask = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].input_mask)
        input_mask = get_input_mask([X_batch, 0])
        return input_mask

    @staticmethod
    def get_output_mask(model, layer, X_batch):
        get_output_mask = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output_mask)
        output_mask = get_output_mask([X_batch, 0])
        return output_mask

    @staticmethod
    def get_input(model, layer, X_batch):
        get_input = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].input)
        _input = get_input([X_batch, 0])
        return _input

    def on_batch_end(self, batch, logs=None):
        # output = self.get_output(self.model, 3, self.training_data[0][:1])
        pass

    def on_epoch_end(self, epoch, logs={}):
        if self.validation_data:
            self.add_predictions(self.validation_data, name="val", logs=logs)
        if self.test_data:
            self.add_predictions(self.test_data, name="test", logs=logs)
        if self.training_data:
            self.add_predictions(self.training_data, name="train", logs=logs)

class PlottingCallback(Callback):
    def __init__(self, benchmarks=None, grid_ranges=None, width=4, height=4):
        super().__init__()
        self.height = height
        self.width = width
        self.benchmarks = benchmarks
        self.grid_ranges = grid_ranges
        self.model_loss = []
        self.validation_loss = []
        self.custom_metrics = defaultdict(list)
        self.fig = None

        models = len(glob.glob(os.path.join(os.path.dirname(__file__), "..", 'results', "model*.png")))
        self.plot_fname = os.path.join(os.path.dirname(__file__), "..", 'results', 'model_{}.png'.format(models + 1))

    def on_train_begin(self, logs={}):
        sns.set_style("whitegrid")
        sns.set_style("whitegrid", {"grid.linewidth": 0.5,
                                    "lines.linewidth": 0.5,
                                    "axes.linewidth": 0.5})
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        sns.set_palette(sns.color_palette(flatui))
        # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        # sns.set_palette(sns.color_palette("Set2", 10))

        plt.ion()  # set plot to animated
        self.fig = plt.figure(
            figsize=(self.width * (1 + len(self.get_metrics(logs))), self.height))  # width, height in inches

        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        geom = mngr.window.geometry()
        x, y, dx, dy = geom.getRect()
        mngr.window.setGeometry(25, 25, dx, dy)

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
        self.fig.set_size_inches(self.width * (1 + len(self.get_metrics(logs))), self.height, forward=True)
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
        for i, (dataset_name, metrics) in enumerate(sorted(custom_metrics_keys.items(), reverse=True)):
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
                axs.plot(self.custom_metrics[entry], label=m, linewidth=linewidth)

            axs.tick_params(labelsize=10)
            labels = list(sorted(metrics))
            if self.benchmarks:
                for (label, benchmark), color in zip(self.benchmarks.items(), ["y", "r"]):
                    axs.axhline(y=benchmark, linewidth=linewidth, color=color)
                    labels = labels + [label]
            axs.legend(labels, loc='upper left', fancybox=True)
            axs.grid(True)
            axs.grid(b=True, which='major', color='gray', linewidth=.5)
            axs.grid(b=True, which='minor', color='gray', linewidth=0.5)

        plt.rcParams.update({'font.size': 10})

        desc = get_model_desc(self.model)
        self.fig.text(.02, .02, desc, verticalalignment='bottom', wrap=True, fontsize=8)
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=.18)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.save_plot()

    def on_train_end(self, logs={}):
        self.save_plot()
