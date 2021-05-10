# ECG-Filtering-AdaptiveFilters
Electrocardiograms (ECG) are devices that record the electric potential that isgenerated from the heart via electrodes placed on the surface of the body. The signal recorded through this method contains random unwanted noise covering the beating of theheart. To better understand and analyze the electrical potential of the heart filters are used to minimize the noise found within the sample to better view the true signal. For thisproject, raw ECG signal data will be taken and subjected to an adaptive filter which is aimed to attenuate the P wave, QRS complex, T wave, and U wave of the ECG signal. Thespecifications will be chosen to best minimize the mean square error and maximize thepeak signal to noise ratio between the raw ECG data and a reference input that iscorrelated with the noise in the primary input. Three adaptive filters were applied: leastmeans square, normalized least means squares, and recursive least squares to comparetheir abilities to most effectively filter the signal.
## Dataset
The ECG signal that was used in this project was obtained using scipy.misc. This libraryhoused an ECG signal that had over 100,000 points. For the purpose of this project the first 600 were selected and acted on.
## Files
### Main
Once the 600 points were extracted from the scipy electrocardiogram, a 2-D array was created ofthe input data. This 2-D input data array was reshaped into a 200x3 matrix and was the firstparameter that was necessary for the padasip filters. Next, a noisy signal was created by adding agenerated noise signal via np.random.normal with a mean of 0 and a standard deviation of 0.2.Once that was completed, a w array of 3x1 random.uniform values from 0 to 1 was created. This would later be used in a target array which was the last parameter necessary for the padasipfilters. This target array was generated by multiplying each w array value to its corresponding column in the reshaped 200x3 input data named signal. When using padasip each filter, LMS, NLMS, and RLS all had the same necessary parameters for the run model under the adaptive filter. Once the run model was successful, it returned three values. The output data, error, and weights. The outputs of each filter were graphed using matlab.plot along with the: starting ECG signal, noise applied ECD signal, reshaped input data, and target. Lastly, the mean square error (MSE) and the peak signal to noise ratio (PSNR) were calculated to determine the efficiency of the model.
