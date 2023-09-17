import h5py
import os
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA, FastICA
from io import BytesIO
import matplotlib.pyplot as plt

class HeartRateExtractorFromThermal:

    def __init__(self, thermal_frames):
        self.thermal_frames = thermal_frames

    @staticmethod
    def read_thermal_data(file_path):
        with h5py.File(file_path, 'r') as f:
            thermal_frames = np.array(f['Frames'])
        return thermal_frames

    #def get_roi(self, frame):
       # return frame[100:200, 100:200]

    def get_roi(self, frame):
        normalized_frame = ((frame - frame.min()) * (255 / (frame.max() - frame.min()))).astype(np.uint8)
    
    # Initialize the face cascade
        face_cascade = cv2.CascadeClassifier('/Users/premkumargudipudi/Documents/main_project/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(normalized_frame, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return frame[y:y+h, x:x+w]
        else:
            return frame
    

    def temporal_signal_extraction(self):
        return [np.mean(self.get_roi(frame)) for frame in self.thermal_frames]

    def bandpass_filter(self, signal, lowcut=0.67, highcut=4.0, fs=50, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def get_heart_rate(self, signal, fs=50):
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        fft_values = np.fft.rfft(signal)
        dominant_frequency = freqs[np.argmax(np.abs(fft_values))]
        return dominant_frequency * 60
    
    def save_combined_plot(self, temporal_signals, transformed_signals, filtered_signals, heart_rates, output_h5_file):
        colors = {
            'FFT': 'r',
            'PCA': 'g',
            'ICA': 'b'
        }
        methods = ['FFT', 'PCA', 'ICA']

        # Temporal Signal Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(temporal_signals['FFT'])  
        ax.set_title("Temporal Signals")
        plt.tight_layout()
        plt.show()

        # Transformed Signal Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in methods:
            ax.plot(transformed_signals[method], color=colors[method], label=method)
        ax.set_title("Transformed Signals (PCA & ICA Components; Temporal for FFT)")
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Filtered Signal Plot with Heart Rate
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in methods:
            label = f"{method}"
            ax.plot(filtered_signals[method], color=colors[method], label=label)
        ax.set_title("Filtered Signals")
        ax.legend()
        plt.tight_layout()
        plt.show()

    

    def method_fft(self):
        temporal_signal = self.temporal_signal_extraction()
        filtered_signal = self.bandpass_filter(temporal_signal)
        heart_rate = self.get_heart_rate(filtered_signal)
        
        return temporal_signal, temporal_signal, filtered_signal, heart_rate

    def method_pca(self):
    
        temporal_signal = self.temporal_signal_extraction()
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        principal_component = principal_components[:, 0]
        filtered_signal = self.bandpass_filter(principal_component)
        heart_rate = self.get_heart_rate(filtered_signal)
        
        return temporal_signal, principal_component, filtered_signal, heart_rate   
            
    def method_ica(self):
        temporal_signal = self.temporal_signal_extraction()
        ica = FastICA(n_components=1, random_state=0)
        independent_components = ica.fit_transform(np.array(temporal_signal).reshape(-1, 1))
        independent_component = independent_components[:, 0]
        filtered_signal = self.bandpass_filter(independent_component)
        heart_rate = self.get_heart_rate(filtered_signal)

        return temporal_signal, independent_component, filtered_signal, heart_rate
