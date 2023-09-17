import h5py
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os
from io import BytesIO
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA

class BreathingRateExtractorFromThermal:

    @staticmethod
    def read_thermal_data(file_path):
        with h5py.File(file_path, 'r') as f:
            thermal_frames = np.array(f['Frames'])
        return thermal_frames

    def __init__(self, thermal_frames):
        self.thermal_frames = thermal_frames

    def get_roi(self, frame):
        h, w = frame.shape
        start_row, end_row = int(h * 0.2), int(h * 0.75)  # Adjusted to cover the upper chest as well
        start_col, end_col = int(w * 0.15), int(w * 0.85)
        return frame[start_row:end_row, start_col:end_col]

    def temporal_signal_extraction(self):
        return [np.mean(self.get_roi(frame)) for frame in self.thermal_frames]



    @staticmethod
    def bandpass_filter(signal, lowcut=0.15, highcut=0.45, fs=50, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)
    @staticmethod
    def get_breathing_rate(signal, fs=50):
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        fft_values = np.fft.rfft(signal)
        dominant_frequency = freqs[np.argmax(np.abs(fft_values))]
        return dominant_frequency * 60
    def method_fft(self):
        temporal_signal = self.temporal_signal_extraction()
        filtered_signal = self.bandpass_filter(temporal_signal)
        breathing_rate = self.get_breathing_rate(filtered_signal)
        return temporal_signal, filtered_signal, breathing_rate

    def method_pca(self):
        temporal_signal = self.temporal_signal_extraction()
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        principal_component = principal_components[:, 0]
        filtered_signal = self.bandpass_filter(principal_component)
        breathing_rate = self.get_breathing_rate(filtered_signal)
        return temporal_signal, principal_component, filtered_signal, breathing_rate

    def method_ica(self):
        temporal_signal = self.temporal_signal_extraction()
        ica = FastICA(n_components=1, random_state=0)
        independent_components = ica.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        independent_component = independent_components[:, 0]
        filtered_signal = self.bandpass_filter(independent_component)
        breathing_rate = self.get_breathing_rate(filtered_signal)
        return temporal_signal, independent_component, filtered_signal, breathing_rate

    def save_plot_to_h5(self, fig, output_h5_file, dset_name="Breathing_Rate_Plot"):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)
        with h5py.File(output_h5_file, 'a') as f:
            if dset_name in f:
                del f[dset_name]
            f.create_dataset(dset_name, data=np.array(bytearray(buf.read())), dtype=np.uint8)
    def visualize_signals(self, output_h5_file=None):
        methods = ['FFT', 'PCA', 'ICA']
        colors = {
            'FFT': 'b',
            'PCA': 'g',
            'ICA': 'r'
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        for method in methods:
            if method == 'FFT':
                temporal_signal, filtered_signal, breathing_rate = self.method_fft()
            elif method == 'PCA':
                temporal_signal, _, filtered_signal, breathing_rate = self.method_pca()
            else:
                temporal_signal, _, filtered_signal, breathing_rate = self.method_ica()

            ax.plot(filtered_signal, color=colors[method], label=f"{method} - Breathing Rate: {breathing_rate:.2f} BR-BPM")

        ax.set_title("Filtered Signals for Breathing Rate Estimation")
        ax.legend()
        
        plt.tight_layout()
        if output_h5_file:
            self.save_plot_to_h5(fig, output_h5_file)
        plt.show()
        
        return breathing_rate
