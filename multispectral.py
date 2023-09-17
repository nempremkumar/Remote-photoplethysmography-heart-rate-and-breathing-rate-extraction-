import h5py
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from io import BytesIO
import os

class HeartRateExtractor:

    def __init__(self, multispectral_frames):
        self.multispectral_frames = multispectral_frames

    @staticmethod
    def read_multispectral_data(file_path):
        with h5py.File(file_path, 'r') as f:
            multispectral_frames = np.array(f['Frames'])
        return multispectral_frames

    @staticmethod
    def extract_bandwidths(frames):
        bandwidth_videos = []
        for i in range(4):
            for j in range(4):
                bandwidth_video = frames[:, i::4, j::4]
                bandwidth_videos.append(bandwidth_video)
        return bandwidth_videos

    def get_roi(self, frame):
        h, w = frame.shape
        start_row, end_row = int(h * 0.25), int(h * 0.75)
        start_col, end_col = int(w * 0.25), int(w * 0.75)
        return frame[start_row:end_row, start_col:end_col]

    def temporal_signal_extraction(self, bandwidth_video):
        return [np.mean(self.get_roi(frame)) for frame in bandwidth_video]

    @staticmethod
    def smooth_signal(signal, window_size=5):
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

    @staticmethod
    def bandpass_filter(signal, lowcut=0.8, highcut=3.0, fs=30, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    @staticmethod
    def get_heart_rate(signal, fs=30):
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        fft_values = np.fft.rfft(signal)
        dominant_frequency = freqs[np.argmax(np.abs(fft_values))]
        return dominant_frequency * 60


    def save_plot(self, plots_data, method_name, output_h5_file, heart_rate, idx):
        fig, ax = plt.subplots(4, 1, figsize=(10, 12))
        
        titles = {
            'FFT': ["Temporal Signal from ROI", "Smoothed Signal", "Bandpass Filtered Signal", "FFT of Filtered Signal"],
            'PCA': ["Temporal Signal from ROI", "Principal Component from PCA", "Smoothed Signal", "Bandpass Filtered Signal"],
            'ICA': ["Temporal Signal from ROI", "Independent Component from ICA", "Smoothed Signal", "Bandpass Filtered Signal"]
        }
        
        for i in range(4):
            ax[i].plot(plots_data[i])
            ax[i].set_title(titles[method_name][i])
            if i == 2:
                ax[i].legend([f"Heart Rate: {heart_rate:.2f} BPM"])
        
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)
        with h5py.File(output_h5_file, 'a') as f:
            dset_name = f"{method_name}_plot_band_{idx}"
            if dset_name in f:
                del f[dset_name]
            f.create_dataset(dset_name, data=np.array(bytearray(buf.read())), dtype=np.uint8)

    def method_fft(self, bandwidth_video, output_h5_file, idx):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        smoothed_signal = self.smooth_signal(temporal_signal)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        
        freqs = np.fft.rfftfreq(len(filtered_signal), 1/30)
        fft_values = np.abs(np.fft.rfft(filtered_signal))
        
        self.save_plot([temporal_signal, smoothed_signal, filtered_signal, fft_values], 'FFT', output_h5_file, heart_rate, idx)
        
        return heart_rate

    def method_pca(self, bandwidth_video, output_h5_file, idx):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        
        N, M, _ = bandwidth_video.shape
        flattened_data = bandwidth_video.reshape(N, M*_)
        pca = PCA(n_components=5)
        principal_components = pca.fit_transform(flattened_data)
        principal_component = principal_components[:, 0]
        smoothed_signal = self.smooth_signal(principal_component)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        
        self.save_plot([temporal_signal, principal_component, smoothed_signal, filtered_signal], 'PCA', output_h5_file, heart_rate, idx)
        
        return heart_rate

    def method_ica(self, bandwidth_video, output_h5_file, idx):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        
        N, M, _ = bandwidth_video.shape
        flattened_data = bandwidth_video.reshape(N, M*_)
        ica = FastICA(n_components=5, random_state=0)
        independent_components = ica.fit_transform(flattened_data)
        independent_component = independent_components[:, 0]
        smoothed_signal = self.smooth_signal(independent_component)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        
        self.save_plot([temporal_signal, independent_component, smoothed_signal, filtered_signal], 'ICA', output_h5_file, heart_rate, idx)
        
        return heart_rate

    