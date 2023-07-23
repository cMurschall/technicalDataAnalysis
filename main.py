import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from setuptools._distutils.command.check import check

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
    fig, axs = plt.subplots(5, 1, figsize=(20, 20), sharex=True)
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


    # set x-axis label
    # format x-axis to show date as day month year
    plt.gcf().axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    # rotate x-axis labels
    plt.gcf().axes[4].tick_params(axis='x', rotation=45)
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
    df = data.read_data(1, 'ADC1')

    # timestamp to datetimeq
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # integrage speed
    df["distance"] = (np.abs(df["speed"] ) * df['time'].diff().dt.total_seconds()).cumsum().fillna(0)



    # cut into chunks
    chunk_length = 20
    chunk_overlap = 0.05


    bins = np.arange(0, df['distance'].max(), chunk_length)
    chunks = []

    # prepare 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.legend()
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_zlabel('Power [dB]')

    # loop over bins
    for i in range(0, len(bins) - 1):
        # get chunk
        chunk = df[(df['distance'] < bins[i + 1])  & (df['distance'] >= bins[i])]

        test_slice = chunk
        fft = np.fft.fft(test_slice['ch0'])
        fft_magnitude = np.abs(fft) / len(test_slice['ch0'])

        # sampling frequency (is supposed to be 20625 Hz )
        fs = 1 / (((test_slice['time'].max() - test_slice['time'].min()).total_seconds()) / len(test_slice['time']))

        ft_freq = np.fft.fftfreq(test_slice['ch0'].size, d=(1/fs))
        idx = np.argsort(ft_freq)

        ax.scatter(i, ft_freq[idx], fft_magnitude[idx])
        # append chunk to list
        chunks.append(chunk)

    plt.show()



    # test_slice = chunks[int(len(chunks) / 2)]
    # fft = np.fft.fft(test_slice['ch0'])
    # fft_magnitude = np.abs(fft) / len(test_slice['ch0'])
#
    # # sampling frequency (is supposed to be 20625 Hz )
    # fs = 1 / (((test_slice['time'].max() - test_slice['time'].min()).total_seconds())/len(test_slice['time']))
#
    # ft_freq = np.fft.fftfreq(test_slice['ch0'].size, d=1/fs)
    # idx = np.argsort(ft_freq)
#
    # power_spectrum = np.abs(np.fft.fft(data))**2
    # plt.plot(ft_freq[idx], fft_magnitude[idx])
    # plt.show()








# if main file is executed
if __name__ == '__main__':
    # print_data_info()

    # plot_positions()
    # plot sensor data
    # plot_sensors_adc(1)
    # plot_sensors_adc(2)
    # plot_imu_sensors()
    cut_into_chunks()



