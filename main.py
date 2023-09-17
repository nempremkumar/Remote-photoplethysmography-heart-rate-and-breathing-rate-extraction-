"""
This project incorporates contributions from various sources:
- A significant portion of the signal extraction algorithms were developed with assistance from Instructor and ChatGPT by OpenAI.
- Subsequent modifications, refinements, and application-specific adjustments were carried out by me to better suit the project's requirements and objectives.

It's essential to recognize that while ChatGPT provided foundational guidance, the final implementation and results were achieved through a collaborative effort, blending external guidance with personal expertise,instructor and project-specific knowledge.
"""


from data_processing import DataProcessor
from multispectral import HeartRateExtractor
from thermal import HeartRateExtractorFromThermal
from breathing_rate import BreathingRateExtractorFromThermal

thermal_file_path = '/Users/premkumargudipudi/Documents/sample_main/20211008_101816_FLIRAX5.h5'
multispectral_file_path = '/Users/premkumargudipudi/Documents/sample_main/20211008_101816_multispec.h5'
output_h5_file = '/Users/premkumargudipudi/Documents/another_main/plots.h5'  # Defined the output path for saving plots
processor = DataProcessor(thermal_file_path, multispectral_file_path)
processor.display_combined_video()

#processor = DataProcessor(thermal_file_path, multispectral_file_path)
#base_directory = "/path/to/base_directory"
#subject_id = "14"
#processor.access_subject_data(base_directory, subject_id)

#frames = HeartRateExtractor.read_multispectral_data(multispectral_file_path)
#bandwidth_videos = HeartRateExtractor.extract_bandwidths(frames)

#extractor = HeartRateExtractor(frames)

#for idx, bw_video in enumerate(bandwidth_videos):
#    heart_rate_fft = extractor.method_fft(bw_video, output_h5_file, idx)
#    print(f"Band {idx} - Estimated Heart Rate using FFT: {heart_rate_fft:.2f} BPM")
    
#    heart_rate_pca = extractor.method_pca(bw_video, output_h5_file, idx)
#    print(f"Band {idx} - Estimated Heart Rate using PCA: {heart_rate_pca:.2f} BPM")
    
#    heart_rate_ica = extractor.method_ica(bw_video, output_h5_file, idx)
#    print(f"Band {idx} - Estimated Heart Rate using ICA: {heart_rate_ica:.2f} BPM")

#thermal_frames = HeartRateExtractorFromThermal.read_thermal_data(thermal_file_path)
#extractor = HeartRateExtractorFromThermal(thermal_frames)


# Using FFT
#heart_rate_fft = extractor.method_fft(output_h5_file)
#print(f"Estimated Heart Rate using FFT: {heart_rate_fft:.2f} BPM")

# Using PCA
#heart_rate_pca = extractor.method_pca(output_h5_file)
#print(f"Estimated Heart Rate using PCA: {heart_rate_pca:.2f} BPM")

# Using ICA
#heart_rate_ica = extractor.method_ica(output_h5_file)
#print(f"Estimated Heart Rate using ICA: {heart_rate_ica:.2f} BPM")

#EXTRACTION OF BREATHING RATE: THERMAL
#thermal_frames = BreathingRateExtractorFromThermal.read_thermal_data(thermal_file_path)
#breathing_extractor = BreathingRateExtractorFromThermal(thermal_frames)
#breathing_rate = breathing_extractor.visualize_signals()
#print(f"Estimated Breathing Rate: {breathing_rate:.2f} BPM")
