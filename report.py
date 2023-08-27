import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Model
from keras import regularizers
from keras.models import load_model

from tqdm.notebook import trange, tqdm

import feature_extraction

import data


def plot_imu_sensors():
    # make a subplot with 5 rows and 1 column
    fig, axs = plt.subplots(6, 1, figsize=(20, 20), sharex=True)

    for j in trange(0, data.count_journeys()):
        df = data.read_data(j, 'IMU')
        # timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        x_color = 'red'
        y_color = 'green'
        z_color = 'blue'
        w_color = 'black'

        for axis in axs:
            axis.axvspan(df['timestamp'].min(), df['timestamp'].max(), facecolor='gray', alpha=0.1)

        axs[0].set_title('Acceleration IMU')
        axs[0].plot(df['timestamp'], df['acc_x'], color=x_color, alpha=.5, )
        axs[0].plot(df['timestamp'], df['acc_y'], color=y_color, alpha=.5, )
        axs[0].plot(df['timestamp'], df['acc_z'], color=z_color, alpha=.5, )

        axs[1].set_title('Gyro IMU')
        axs[1].plot(df['timestamp'], df['gyro_x'], color=x_color, alpha=.5)
        axs[1].plot(df['timestamp'], df['gyro_y'], color=y_color, alpha=.5)
        axs[1].plot(df['timestamp'], df['gyro_y'], color=z_color, alpha=.5)

        axs[2].set_title('Velocity IMU')
        axs[2].plot(df['timestamp'], df['ang_vel_x'], color=x_color, alpha=.5)
        axs[2].plot(df['timestamp'], df['ang_vel_y'], color=y_color, alpha=.5)
        axs[2].plot(df['timestamp'], df['ang_vel_z'], color=z_color, alpha=.5)

        axs[3].set_title('Orientation IMU')
        axs[3].plot(df['timestamp'], df['orientation_x'], color=x_color, alpha=.5)
        axs[3].plot(df['timestamp'], df['orientation_y'], color=y_color, alpha=.5)
        axs[3].plot(df['timestamp'], df['orientation_z'], color=z_color, alpha=.5)
        axs[3].plot(df['timestamp'], df['orientation_w'], color=w_color, alpha=.5)

        axs[4].set_title('Pitch IMU')
        axs[4].plot(df['timestamp'], df['pitch'], color="black")

        axs[5].set_title('Roll IMU')
        axs[5].plot(df['timestamp'], df['roll'], color="black")

    # set x-axis label
    # format x-axis to show date as day month year
    plt.gcf().axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    # rotate x-axis labels
    plt.gcf().axes[5].tick_params(axis='x', rotation=45)
    # set margin to avoid cut-off of labels
    plt.gcf().subplots_adjust(bottom=0.25)

    # tight layout
    plt.tight_layout()
    return plt


def plot_sensors_adc(adc=1):
    # make a subplot with 5 rows and 1 column
    fig, axs = plt.subplots(6, 1, figsize=(20, 20), sharex=True)
    for j in trange(0, data.count_journeys()):
        df = data.read_data(j, 'ADC' + str(adc))
        # timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # convert speed to km/h
        df['speed'] = df['speed'] * 3.6

        every_nth = 60_000

        for axis in axs:
            axis.axvspan(df['time'].min(), df['time'].max(), facecolor='gray', alpha=0.1)

        axs[0].set_title('Ch0 - ADC' + str(adc))
        axs[0].plot(df['time'][::every_nth], df['ch0'][::every_nth], color='red')

        axs[1].set_title('Ch1 - ADC' + str(adc))
        axs[1].plot(df['time'][::every_nth], df['ch1'][::every_nth], color='green')

        axs[2].set_title('Ch2 - ADC' + str(adc))
        axs[2].plot(df['time'][::every_nth], df['ch2'][::every_nth], color='blue')

        axs[3].set_title('Ch3 - ADC' + str(adc))
        axs[3].plot(df['time'][::every_nth], df['ch3'][::every_nth], color='black')

        axs[4].set_title('Speed - ADC' + str(adc))
        axs[4].plot(df['time'][::every_nth], df['speed'][::every_nth], color='orange')

        axs[5].set_title('Profile - ADC' + str(adc))
        axs[5].plot(df['time'][::every_nth], df['ch0_local_distance'][::every_nth], color='orange')

    # set x-axis label
    # format x-axis to show date as day month year
    plt.gcf().axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    # rotate x-axis labels
    plt.gcf().axes[5].tick_params(axis='x', rotation=45)
    # set margin to avoid cut-off of labels
    plt.gcf().subplots_adjust(bottom=0.25)

    # tight layout
    plt.tight_layout()
    return plt


def plot_positions():
    fig, axs = plt.subplots(5, 4, figsize=(20, 20), sharex=True, sharey=True, dpi=200)
    axs = np.concatenate(axs)
    for j in trange(0, data.count_journeys()):
        df_imu = data.read_data(j, 'IMU')
        # timestamp to datetime
        df_imu['timestamp'] = pd.to_datetime(df_imu['timestamp'], unit='s')
        df_imu["vel_x"] = df_imu["acc_x"] * df_imu['timestamp'].diff().dt.total_seconds()
        df_imu["pos_x"] = (df_imu["vel_x"] * df_imu['timestamp'].diff().dt.total_seconds()).cumsum()

        df_imu["vel_y"] = df_imu["acc_y"] * df_imu['timestamp'].diff().dt.total_seconds()
        df_imu["pos_y"] = (df_imu["vel_y"] * df_imu['timestamp'].diff().dt.total_seconds()).cumsum()

        # calculate distance from x and y
        df_imu["distance"] = np.sqrt(df_imu["pos_x"] ** 2 + df_imu["pos_y"] ** 2)
        # print(f"Journey {j} distance: {(df_imu['distance'].max()) :.2f} m")

        axs[j].scatter(df_imu['pos_x'], df_imu['pos_y'], color='red', s=1)
        # plot distance over time
        # axs[j].plot(df_imu['timestamp'], df_imu['distance'], color='blue', alpha=.5)
        axs[j].set_title(f"J {j + 1} ({df_imu['distance'].max() :.0f} m)")

    # clear last two plots
    axs[-1].axis('off')
    axs[-2].axis('off')
    return plt


def plot_frequencies_in_bins(sources=['ADC1', 'ADC2'], chunk_length=0.2, chunk_overlap=0):
    # make matplotlib color array with data.count_journeys() colors
    colors = cm.rainbow(np.linspace(0, 1, data.count_journeys()))

    # make a subplot with 2 rows and 1 column
    fig, axs = plt.subplots(nrows=1, ncols=len(sources), figsize=(15, 5), sharey=True)

    for s, source in enumerate(sources):
        end_count = data.count_journeys()

        for j in (pbar := trange(0, end_count)):
            pbar.set_description(f"Source {source}, Journey {j}")

            # print(f"Journey {j}")
            df = data.read_data(j, source)

            bins = np.arange(0, df['distance'].max(), chunk_length)

            # loop over bins
            for i in range(0, len(bins) - 1):
                # get chunk with overlap
                chunk = df[(df['distance'] < bins[i + 1] + chunk_overlap) & (df['distance'] >= bins[i] - chunk_overlap)]

                fft = np.fft.rfft(chunk['ch0'])
                fft_magnitude = np.abs(fft) / len(chunk['ch0'])

                # sampling frequency (is supposed to be 20625 Hz )
                fs = 1 / (((chunk['time'].max() - chunk['time'].min()).total_seconds()) / len(chunk['time']))

                ft_freq = np.fft.rfftfreq(chunk['ch0'].size, d=(1 / fs))
                idx = np.argsort(ft_freq)

                axs[s].plot(ft_freq[idx], fft_magnitude[idx], c=colors[j], alpha=.2)

        axs[s].set_title(source)
        axs[s].set_xlabel('Frequency [Hz]')
        axs[s].set_ylabel('Magnitude')

    fig.tight_layout()
    return plt


def conduct_pca_on_featues(source='ADC1', count_components=2, feature_lambda=None):
    # cfg_file = tsfel.get_features_by_domain(json_path='features.json')
    colors = cm.rainbow(np.linspace(0, 1, data.count_journeys()))

    fig, axes = plt.subplots(nrows=count_components, ncols=count_components)

    for j in trange(0, data.count_journeys(), desc="Journey"):
        # print(f"Journey {j}")
        df = data.read_as_chuncked(j, source)

        features = np.empty((0, feature_extraction.get_feature_size()[0]))

        # loop over bins
        for i in range(0, max(df['Bin'])):
            # get bins signal
            signal = df[df['Bin'] == i]['ch0'].values

            if feature_lambda is not None:
                signal = feature_lambda(signal)

            # signal = np.abs(np.fft.fft(signal))

            f = feature_extraction.extract(signal)
            features = np.vstack((features, f))

        # scale our data
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # calculate pca from features
        pca = PCA(n_components=count_components)

        try:

            features_pca = pca.fit_transform(features)
            # print pca explained variance in percent
            # Using set_printoptions
            np.set_printoptions(suppress=True)
            print(
                f"Journey {j + 1} explained variances: {pca.explained_variance_ratio_}, total: {np.sum(pca.explained_variance_ratio_):.2%}")

            for row in range(axes.shape[0]):
                for col in range(axes.shape[1]):
                    ax = axes[row, col]
                    if row == col:
                        ax.tick_params(
                            axis='both', which='both',
                            bottom='off', top='off',
                            labelbottom='off',
                            left='off', right='off',
                            labelleft='off')
                        ax.text(0.5, 0.5, f"PCA component {col + 1}", horizontalalignment='center')
                        # remove ticks
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax.scatter(features_pca[:, row], features_pca[:, col], color=colors[j], s=2, alpha=0.5)
                        ax.scatter(features_pca[:, row], features_pca[:, col], color=colors[j], s=2, alpha=0.5)
                        ax.scatter(features_pca[:, row], features_pca[:, col], color=colors[j], s=2, alpha=0.5)

            # plot pca
            # for c in range(1, count_components):
            #    plt.scatter(features_pca[:, 0], features_pca[:, c], color=colors[j], s=2, alpha=0.5)

            # plt.xlabel('PCA 0')
            # plt.ylabel(f"PCA 1-{count_components}")

            # plt.scatter(features_pca[:, 0], features_pca[:, 1], color=colors[j], s=2, alpha=0.3)
            # plt.scatter(features_pca[:, 0], features_pca[:, 2], color=colors[j], s=2, alpha=0.3)
            # plt.scatter(features_pca[:, 0], features_pca[:, 3], color='blue', s=2, alpha=0.3)
            # plt.scatter(features_pca[:, 0], features_pca[:, 4], color='yellow', s=2, alpha=0.3)
        except Exception as e:
            print(e)

    fig.tight_layout()
    return plt


def train_lstm_autoencoder(model_version, features=['speed', 'ch0_fft']):
    """
    Train a LSTM autoencoder model.
    inspired by this paper: https://arxiv.org/pdf/2101.11539.pdf
    Parameters
    ----------
    model_version : int
    features : list

    Returns
    -------

    """

    def make_model_1(train_data_shape):
        model = Sequential(name="model_1")
        model.add(LSTM(128, input_shape=(train_data_shape[1], train_data_shape[2])))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(train_data_shape[1]))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(train_data_shape[2])))
        model.compile(optimizer='adam', loss='mae')
        model.summary()
        return model

    # Try another model
    def make_model_2(train_data_shape):
        model = Sequential(name="model_2")
        model.add(LSTM(16, input_shape=(train_data_shape[1], train_data_shape[2]),
                       activation='relu',
                       return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.02)))

        model.add(LSTM(4, activation='relu', return_sequences=False))
        model.add(RepeatVector(train_data_shape[1]))
        model.add(LSTM(4, return_sequences=True))
        model.add(LSTM(16, return_sequences=True))
        model.add(TimeDistributed(Dense(train_data_shape[2])))
        model.compile(optimizer='adam', loss='mae')
        model.summary()
        return model

    # Try jet another model
    def make_model_3(train_data_shape):
        count_neurons = 8
        model = Sequential(name="model_3")
        model.add(LSTM(count_neurons, input_shape=(train_data_shape[1], train_data_shape[2]), return_sequences=True))
        model.add(LSTM(int(count_neurons / 2), return_sequences=False))
        model.add(RepeatVector(train_data_shape[1]))
        model.add(LSTM(int(count_neurons / 2), return_sequences=True))
        model.add(LSTM(count_neurons, return_sequences=True))
        model.add(TimeDistributed(Dense(train_data_shape[2])))

        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def make_model(train_data_shape):
        model_file_name = f"model_{model_version}.keras"
        if os.path.exists(model_file_name):
            m = tf.keras.models.load_model(model_file_name)
            return m

        if model_version == 1:
            return make_model_1(train_data_shape)
        if model_version == 2:
            return make_model_2(train_data_shape)
        if model_version == 3:
            return make_model_3(train_data_shape)

    with tf.device('/gpu:0'):
        source = 'ADC1'

        # modelshape => [samples, timesteps, features]
        model = make_model((None, 1, len(features)))

        print(f"Caluclating Scaler for all journeys for source {source}.")
        scaler = data.create_scaler_for_features(source, features)
        print("Scaler calculated.")

        df_loss = pd.DataFrame(columns=['Journey', 'Bin', 'Loss'])
        df_journey_errors = pd.DataFrame(columns=['Journey', 'MeanAbsoluteError'])

        # we skip those journeys and use them later to test against.
        count_journeys_test = 2
        count_journeys = data.count_journeys()
        for journey in range(count_journeys_test, count_journeys):
            df_journey = data.read_as_chuncked(journey, source)
            df_journey["speed"] = np.abs(df_journey["speed"])

            # we get all data for this journey
            print(
                f"Training on journey {journey}/{count_journeys}. Distance: {df_journey['distance'].max():.2f} m, Max "
                f"speed: {df_journey['speed'].max():.2} m/s, Dura"
                f"tion: {(df_journey['time'].max() - df_journey['time'].min()).total_seconds() / 60:.1f} min")

            # scale intput data to our current data
            features_transformed = scaler.transform(df_journey[features])

            count_bins = df_journey["Bin"].max()
            for training_bin in range(count_bins):

                # get training data for this bin
                training_data = features_transformed[df_journey['Bin'] == training_bin]

                # we use 1% of the data as sequence size
                seq_size = int(len(training_data) / 100)  # Number of time steps to look back

                # create generator => this seems to be deprecated, but is just do memory efficient
                generator = TimeseriesGenerator(training_data, training_data, length=seq_size, batch_size=100)

                # training the model. We use 5 epochs, this is not much, but we have a ton of data, so we don't need more
                history = model.fit(generator, epochs=5, verbose=0)

                loss = history.history['loss'][-1]
                if training_bin % int(count_bins / 5) == 0:
                    print(f"   Trained bin {training_bin}/{count_bins} with sequence size {seq_size}, loss: {loss:.4f}")

                new_df = pd.DataFrame([{'Journey': journey, 'Bin': training_bin, 'Loss': loss}])
                df_loss = pd.concat([df_loss, new_df], ignore_index=True)

            # lets evaluate the model for this journey
            journey_data = features_transformed.reshape(-1, 1, 2)

            # this might take some time..
            trainPredict = model.predict(journey_data, verbose=2)
            train_mean_absolute_error_distance = np.mean(np.abs(trainPredict - journey_data), axis=1)[:, 1]
            df_journey_errors = pd.concat([df_journey_errors, pd.DataFrame([{
                'Journey': journey,
                'MeanAbsoluteError': [train_mean_absolute_error_distance]
            }])], ignore_index=True)

            # store our good states
            pd.to_pickle(df_journey_errors, f"errors_{model.name}.pkl")
            pd.to_pickle(df_loss, f"loss_{model.name}.pkl")
            model.save(f"{model.name}.keras")


def evaluate_lstm_autoencoder(model_version, features=['speed', 'ch0'], percentil_threshold=95):
    source = 'ADC1'

    # load model
    model = tf.keras.models.load_model(f"./results/model_{model_version}/model.keras")

    # print model summary
    print("Model summary:")
    model.summary()
    # errors_df = pd.read_pickle(f"./results/model_{model_version}/errors.pkl")
    loss_df = pd.read_pickle(f"./results/model_{model_version}/loss.pkl")

    fig, axs = plt.subplots()
    axs.plot(loss_df["Loss"], 'r-', label="Loss", linewidth=0.5, alpha=0.5)

    print("Plot training loss")
    for j in loss_df['Journey'].unique():
        rect_x = loss_df[loss_df['Journey'] == j].index[0]
        rect_y = loss_df[loss_df['Journey'] == j].index[-1]
        axs.axvspan(rect_x, rect_y, facecolor='gray', edgecolor="black", alpha=0.1, linewidth=1)

        # write centered text
        text_x = loss_df[loss_df['Journey'] == j].index[0] + (
                loss_df[loss_df['Journey'] == j].index[-1] - loss_df[loss_df['Journey'] == j].index[0]) / 2
        text_y = np.max(loss_df["Loss"]) * 0.8
        axs.text(text_x, text_y, f"Journey {j + 1}", verticalalignment='top', horizontalalignment='center', fontsize=8,
                 rotation=45)

    axs.set_xlabel("Trained Bin")
    axs.set_ylabel("Training Loss")

    fig.suptitle(f"Training Loss of Model {model_version}")

    plt.show()

    scaler = data.create_scaler_for_features(source, features)
    errors = []

    print("Calculate loss of training to determine the threshold for anomalies.")
    for j in loss_df['Journey'].unique():

        df = data.read_as_chuncked(j, source)

        # transform the features by the same scaler we used for training
        df_transformed = scaler.transform(df[features]).reshape(-1, 1, 2)

        # we predict the reconstruction
        prediction = model.predict(df_transformed)

        # we calculate reconstruction error (mean absolute error)
        mean_absolute_error = np.mean(np.abs(prediction - df_transformed), axis=1)[:, 1]

        print(mean_absolute_error.shape)
        errors.append(mean_absolute_error)

    # we flatten the list of errors
    training_errors = np.concatenate(errors)

    # we use the giben percentile as threshold
    max_train_error = np.percentile(training_errors, percentil_threshold)


    # plot error distribution
    plt.title('Error Distribution')
    plt.yscale("log")

    hist = plt.hist(training_errors, bins=30, density=True, label="reconstruction error", color="#D3D3D3", width=0.7)

    # we limit the hight of the threshold line, so it does not look too aggressive
    y_ver_max = hist[0][0] * 0.8
    plt.axvline(x=max_train_error, ymax=y_ver_max, color='r', label="reconstruction threshold")

    plt.ylabel("Mean Absolute Error (log)")
    plt.xlabel("Reconstruction Error")

    plt.legend()
    plt.show()

    scaler = data.create_scaler_for_features(source, features)

    count_test_journeys = 3


    _, axs = plt.subplots(count_test_journeys, figsize=(15, 15))


    for j in range(count_test_journeys):
        df_journey = data.read_data(j, source)
        df_journey["speed"] = np.abs(df_journey["speed"])

        # scale intput data to all our recorded data
        features_transformed = scaler.transform(df_journey[features])
        # lets evaluate the model for this journey
        journey_data = features_transformed.reshape(-1, 1, 2)
        # this might take some time..
        test_predict = model.predict(journey_data)

        test_error = np.mean(np.abs(test_predict - journey_data), axis=1)

        # again we only look for the errors of the profile reconstruction
        anomalies = test_error[:, 1] >= max_train_error

        to_mm = 10 ** 7
        axs[j].set_title(f"Journey {j} - {np.sum(anomalies)} anomalies found")

        # reset scale
        axs[j].set_yscale("linear")

        axs[j].scatter(df_journey[anomalies]["distance"], df_journey[anomalies]["ch0"] * to_mm, color="red", s=1.5)
        axs[j].plot(df_journey["distance"], df_journey["ch0"] * to_mm, color="#D3D3D3", linewidth=0.2)


    for ax in axs.flat:
        ax.set(xlabel='Track distance [m]', ylabel='Profile in [mm]')

    plt.tight_layout()
    plt.show()


### Helpers
def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])
