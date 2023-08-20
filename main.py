import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import tsfel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Model
from keras import regularizers

import feature_extraction

import data


# plt.rcParams.update({'font.size': 30})


def print_data_info():
    """
    Print information about the data.\
    :return:
    """
    for j in range(0, data.count_journeys()):
        print('Journey ' + str(j) + ':')
        for _, sensor in enumerate(data.get_sensors()):
            df = data.read_data(j, sensor)
            print('  ' + sensor + ': ' + str(len(df)) + ' samples')


def plot_imu_sensors():
    # make a subplot with 5 rows and 1 column
    fig, axs = plt.subplots(6, 1, figsize=(20, 20), sharex=True)

    for j in range(0, data.count_journeys()):
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
    plt.show()
    plt.savefig('imu_sensors.png')


def plot_sensors_adc(adc=1):
    # make a subplot with 5 rows and 1 column
    fig, axs = plt.subplots(6, 1, figsize=(20, 20), sharex=True)
    for j in range(0, data.count_journeys()):
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
    plt.show()


def plot_positions():
    fig, axs = plt.subplots(5, 4, figsize=(20, 20), sharex=True, sharey=True, dpi=200)
    axs = np.concatenate(axs)
    for j in range(0, data.count_journeys()):
        df_imu = data.read_data(j, 'IMU')
        # timestamp to datetime
        df_imu['timestamp'] = pd.to_datetime(df_imu['timestamp'], unit='s')
        df_imu["vel_x"] = df_imu["acc_x"] * df_imu['timestamp'].diff().dt.total_seconds()
        df_imu["pos_x"] = (df_imu["vel_x"] * df_imu['timestamp'].diff().dt.total_seconds()).cumsum()

        df_imu["vel_y"] = df_imu["acc_y"] * df_imu['timestamp'].diff().dt.total_seconds()
        df_imu["pos_y"] = (df_imu["vel_y"] * df_imu['timestamp'].diff().dt.total_seconds()).cumsum()

        # calculate distance from x and y
        df_imu["distance"] = np.sqrt(df_imu["pos_x"] ** 2 + df_imu["pos_y"] ** 2)
        print(f"Journey {j} distance: {(df_imu['distance'].max()) :.2f} m")

        axs[j].scatter(df_imu['pos_x'], df_imu['pos_y'], color='red', s=1)
        # plot distance over time
        # axs[j].plot(df_imu['timestamp'], df_imu['distance'], color='blue', alpha=.5)
        axs[j].set_title(f"J {j + 1} ({df_imu['distance'].max() :.0f} m)")

    plt.show()


def cut_into_chunks():
    # make matplotlib color array with data.count_journeys() colors
    colors = cm.rainbow(np.linspace(0, 1, data.count_journeys()))

    for j in range(0, data.count_journeys()):
        print(f"Journey {j}")
        df = data.read_data(j, 'ADC1')

        # timestamp to datetimeq
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # integrage speed
        df["distance"] = (np.abs(df["speed"]) * df['time'].diff().dt.total_seconds()).cumsum().fillna(0)

        # cut into chunks
        chunk_length = .2
        chunk_overlap = 0.05

        bins = np.arange(0, df['distance'].max(), chunk_length)
        chunks = []

        # prepare 3d plot
        # loop over bins
        for i in range(0, len(bins) - 1):
            # get chunk
            chunk = df[(df['distance'] < bins[i + 1]) & (df['distance'] >= bins[i])]

            fft = np.fft.rfft(chunk['ch0'])
            fft_magnitude = np.abs(fft) / len(chunk['ch0'])

            # sampling frequency (is supposed to be 20625 Hz )
            fs = 1 / (((chunk['time'].max() - chunk['time'].min()).total_seconds()) / len(chunk['time']))

            ft_freq = np.fft.rfftfreq(chunk['ch0'].size, d=(1 / fs))
            idx = np.argsort(ft_freq)

            plt.plot(ft_freq[idx], fft_magnitude[idx], c=colors[j], alpha=.2)
            # append chunk to list
            chunks.append(chunk)

    # set x label
    plt.xlabel('Frequency [Hz]')
    # set y label
    plt.ylabel('Magnitude')

    plt.show()


def cut_into_feature_chunks():
    # make matplotlib color array with data.count_journeys() colors
    colors = cm.rainbow(np.linspace(0, 1, data.count_journeys()))
    cfg_file = tsfel.get_features_by_domain(domain='temporal')

    for j in range(0, data.count_journeys()):
        print(f"Journey {j}")
        df = data.read_data(j, 'ADC1')

        # timestamp to datetimeq
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # integrage speed
        df["distance"] = (np.abs(df["speed"]) * df['time'].diff().dt.total_seconds()).cumsum().fillna(0)

        # cut into chunks
        chunk_length = 0.1
        chunk_overlap = 0.05

        bins = np.arange(0, df['distance'].max(), chunk_length)
        chunks = []

        # prepare 3d plot
        # loop over bins
        for i in range(0, len(bins) - 1):
            # get chunk
            chunk = df[(df['distance'] < bins[i + 1]) & (df['distance'] >= bins[i])].copy()

            features = tsfel.time_series_features_extractor(cfg_file, chunk['ch0'], window_size=1, fs=20625, verbose=0)
            print(features.size)


def find_features():
    cfg_file = tsfel.get_features_by_domain(json_path='features.json')
    colors = cm.rainbow(np.linspace(0, 1, data.count_journeys()))

    for j in range(0, data.count_journeys()):
        print(f"Journey {j}")
        df = data.read_data(j, 'ADC1')

        features = np.empty((0, feature_extraction.get_feature_size()[0]))
        chunk_length = 5
        bins = np.arange(0, df['distance'].max(), chunk_length)

        # loop over bins
        for i in range(0, len(bins) - 1):
            # get chunk
            chunk = df[(df['distance'] < bins[i + 1]) & (df['distance'] >= bins[i])].copy()

            signal = chunk['ch0'].values

            signal = np.abs(np.fft.fft(signal))

            f = feature_extraction.extract(signal)
            features = np.vstack((features, f))

        # scale our data
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # calculate pca from features
        pca = PCA(n_components=5)

        try:

            features_pca = pca.fit_transform(features)
            # print pca explained variance in percent
            # Using set_printoptions
            np.set_printoptions(suppress=True)
            print(
                f"explained variances: {pca.explained_variance_ratio_}, total: {np.sum(pca.explained_variance_ratio_):.4f}")

            # plot pca
            plt.scatter(features_pca[:, 0], features_pca[:, 1], color=colors[j], s=2, alpha=0.3)
            plt.scatter(features_pca[:, 0], features_pca[:, 2], color=colors[j], s=2, alpha=0.3)
            # plt.scatter(features_pca[:, 0], features_pca[:, 3], color='blue', s=2, alpha=0.3)
            # plt.scatter(features_pca[:, 0], features_pca[:, 4], color='yellow', s=2, alpha=0.3)
        except:
            print("error")
    plt.show()


def use_lstm_autoencoder():
    def make_model(train_data_shape):
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_data_shape[1], train_data_shape[2])))
        model.add(Dropout(rate=0.2))

        model.add(RepeatVector(train_data_shape[1]))

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(train_data_shape[2])))
        model.compile(optimizer='adam', loss='mae')
        model.summary()
        return model

    # [samples, timesteps, features]
    features = ['speed', 'ch0_local_distance']
    model = make_model((None, 1, len(features)))

    adc1_data = data.read_as_chuncked(source='ADC1')

    adc1_data["speed"] = np.abs(adc1_data["speed"])

    # adc_2 = data.read_as_chuncks(source='ADC2')

    ct = ColumnTransformer([
        ('somename', StandardScaler(), features)
    ], remainder='passthrough')


    scaler = ct.fit(adc1_data[features])

    df_loss = pd.DataFrame(columns=['Journey', 'Bin', 'Loss'])

    count_journeys = int(adc1_data['Journey'].max())
    for journey in range(1):
        print(f"Training on journey {journey}/{count_journeys}")

        df_journey = adc1_data[adc1_data['Journey'] == journey]

        # scale
        features_transformed = scaler.transform(df_journey[features])

        count_bins = df_journey["Bin"].max()
        for training_bin in range(count_bins):
            train_x = features_transformed[df_journey['Bin'] == training_bin]
            # df_bin = df_journey[df_journey['Bin'] == training_bin]

            seq_size = int(len(train_x) / 100)  # Number of time steps to look back
            print(f"Training on bin {training_bin}/{count_bins} with sequence size {seq_size}")

            generator = TimeseriesGenerator(train_x, train_x, length=seq_size, batch_size=100)
            history = model.fit(generator, epochs=5, verbose=2)

            new_df = pd.DataFrame([{'Journey': journey, 'Bin': training_bin, 'Loss': history.history['loss'][-1]}])
            df_loss = pd.concat([df_loss, new_df], ignore_index=True)

        df_loss.to_csv('loss.csv')
        model.save('model.h5')






# if main file is executed
if __name__ == '__main__':
    # print_data_info()

    # plot_positions()
    # plot sensor data
    # plot_sensors_adc(1)
    # plot_sensors_adc(2)
    # plot_imu_sensors()
    # find_features()

    use_lstm_autoencoder()
