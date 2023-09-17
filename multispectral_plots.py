import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA

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
    def bandpass_filter(signal, lowcut=0.8, highcut=4.0, fs=30, order=1):
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


    def method_fft_multispectral(self, bandwidth_video):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        smoothed_signal = self.smooth_signal(temporal_signal)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        return temporal_signal, smoothed_signal, filtered_signal, heart_rate

    
    def method_pca_multispectral(self, bandwidth_video):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        principal_component = principal_components[:, 0]
        principal_components = pca.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        principal_component = principal_components[:, 0]

        smoothed_signal = self.smooth_signal(principal_component)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        return temporal_signal, principal_component, smoothed_signal, filtered_signal, heart_rate

    def method_ica_multispectral(self, bandwidth_video):
        temporal_signal = self.temporal_signal_extraction(bandwidth_video)
        ica = FastICA(n_components=5, random_state=0)
        independent_components = ica.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        independent_component = independent_components[:, 0]
        independent_components = ica.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        independent_component = independent_components[:, 0]

        smoothed_signal = self.smooth_signal(independent_component)
        filtered_signal = self.bandpass_filter(smoothed_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        return temporal_signal, independent_component, smoothed_signal, filtered_signal, heart_rate


# ... [rest of the import statements and HeartRateExtractor class methods remain unchanged]

    def plot_multispectral_data(self, data, title_prefix, output_file):
        plt.figure(figsize=(12, 8))
        for idx, (band, signals) in enumerate(data["FFT"].items()):
            plt.plot(signals[0], label=band)
        plt.title(f"{title_prefix} for all bandwidths")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file + "_AllBands.png")
        plt.show()

    def plot_transformed_data(self, data, title_prefix, output_file):
        methods = ["FFT", "PCA", "ICA"]
        for method in methods:
            plt.figure(figsize=(12, 8))
            for idx, (band, signals) in enumerate(data[method].items()):
                plt.plot(signals[1], label=band)
            plt.title(f"{title_prefix} using {method} for all bandwidths")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_file + f"_{method}.png")
            plt.show()

    def plot_filtered_data(self, data, title_prefix, output_file):
        methods = ["FFT", "PCA", "ICA"]
        for method in methods:
            plt.figure(figsize=(12, 8))
            for idx, (band, signals) in enumerate(data[method].items()):
                # Extract the heart rate value
                heart_rate = signals[3][0] if isinstance(signals[3], np.ndarray) else signals[3]
                plt.plot(signals[2], label=f"{band}- {heart_rate:.2f} BPM")
            plt.title(f"{title_prefix} using {method} for all bandwidths")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_file + f"_{method}.png")
            plt.show()

# MULTISPECTRAL PROCESSING
multispectral_frames = HeartRateExtractor.read_multispectral_data(multispectral_file_path)
multispectral_extractor = HeartRateExtractor(multispectral_frames)

all_data = {
    "FFT": {},
    "PCA": {},
    "ICA": {}
}
bandwidth_videos = multispectral_extractor.extract_bandwidths(multispectral_frames)
for idx, bandwidth_video in enumerate(bandwidth_videos):
    all_data["FFT"][f"band_{idx}"] = multispectral_extractor.method_fft_multispectral(bandwidth_video)
    all_data["PCA"][f"band_{idx}"] = multispectral_extractor.method_pca_multispectral(bandwidth_video)
    all_data["ICA"][f"band_{idx}"] = multispectral_extractor.method_ica_multispectral(bandwidth_video)

# Saving the plots
multispectral_extractor.plot_multispectral_data(all_data, "Temporal Signal", "TemporalSignal")
multispectral_extractor.plot_transformed_data(all_data, "Transformed Signal", "TransformedSignal")
multispectral_extractor.plot_filtered_data(all_data, "Filtered Signal", "FilteredSignal")



