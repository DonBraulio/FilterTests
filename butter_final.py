from scipy.signal import butter, lfilter, square
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy import interpolate


# Bandpass filter implemented with coefficients calculated
# as 2nd order butterworth with poles in 0.1Hz and 25Hz,
# fs = 100Hz
def final_implementation_100hz(u):
    y = [0.0]*len(u)
    for k in range(0, len(u)):
        u_2 = u[k-2] if k > 1 else 0
        y_2 = y[k-2] if k > 1 else 0
        y_1 = y[k-1] if k > 0 else 0
        y[k] = 0.4984292*(u[k] - u_2) + 0.9968584 * y_1 - 0.0031416 * y_2
    return y


# Bandpass filter implemented with coefficients calculated
# as 2nd order butterworth with poles in 0.1Hz and 25Hz,
# fs = 400Hz
def final_implementation_400hz(u):
    y = [0.0]*len(u)
    for k in range(0, len(u)):
        u_2 = u[k-2] if k > 1 else 0
        y_2 = y[k-2] if k > 1 else 0
        y_1 = y[k-1] if k > 0 else 0
        y[k] = 0.16534236*(u[k] - u_2) + 1.66879379 * y_1 - 0.66931528 * y_2
    return y


# Calculate Z-transform numerator (b) and denominator (a)
# for a butterworth bandpass filter with specified cutoff freqs
def create_butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Calculate filter using create_butter_bandpass()
# and apply it directly to the input data
def apply_butter_bandpass(data, lowcut, highcut, fs, order=5):
    b, a = create_butter_bandpass(lowcut, highcut, fs, order=order)
    print("Numerator for order %d (fs = %dHz): " % (order, fs), b)
    print("Denominator for order %d (fs = %dHz): " % (order, fs), a)
    y = lfilter(b, a, data)
    return y


# --------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Sample rate and desired cutoff frequencies (in Hz).
    fs1 = 100.0  # Normal Sample rate (100Hz)
    fs2 = 400.0  # Sample rate for diagnostic mode (400Hz)
    lowcut = 0.1  # Low freq. pole of bandpass filter
    highcut = 25.0  # High freq. pole

    # ----------------- FREQ RESPONSE FOR DIFFERENT FILTERS -----------------
    # Plot freq response for different filter orders and fs
    plt.close('all')
    f, (mag_plot, phase_plot) = plt.subplots(2, sharex=True)
    for fs_i, order in [(fs1, 1), (fs1, 2), (fs2, 1)]:
        b, a = create_butter_bandpass(lowcut, highcut, fs_i, order=order)
        w, h = freqz(b, a, worN=5000)
        mag_plot.semilogx((fs_i / (2*np.pi)) * w, abs(h), label="order = %d, fs = %d" % (2 * order, fs_i))
        phase_plot.semilogx((fs_i / (2*np.pi)) * w, np.angle(h) * 180 / np.pi,
                            label="order = %d, fs = %d" % (2 * order, fs_i))
    mag_plot.set_title('Gain')
    phase_plot.set_title('Phase (deg)')
    mag_plot.grid(True)
    phase_plot.grid(True)
    mag_plot.legend(loc='best')
    phase_plot.legend(loc='best')
    plt.xlabel('Frequency (Hz)')
    plt.show()

    # ---------------------------- INPUT SIGNAL ----------------------------
    # Generate input signal with:
    #  - DC offset
    #  - One square pulse
    #  - Pulses of various frequencies (with exponential envelope waves)
    #  - Noise (higher frequency sinusoidals)
    T = 10.0
    nsamples = T * fs1
    t = np.linspace(0, T, nsamples, endpoint=False)
    # DC offset
    input_signal = 2
    # Sinusoidal pulses (exponential envelopes)
    a = 1  # Pulses amplitude
    input_signal += a * np.exp(-20 * (t - 1 * T / 8) ** 2) * np.cos(2 * np.pi * 5.0 * t)  # Passband wave
    input_signal += a * np.exp(-20 * (t - 2 * T / 8) ** 2) * np.cos(2 * np.pi * 10.0 * t)  # Passband wave
    input_signal += a * np.exp(-20 * (t - 3 * T / 8) ** 2) * np.cos(2 * np.pi * 15.0 * t)  # Passband wave
    input_signal += a * np.exp(-20 * (t - 4 * T / 8) ** 2) * np.cos(2 * np.pi * 25.0 * t)  # Pole frequency
    # x += a * np.exp(-20 * (t - 3*T/8)**2) * np.cos(2 * np.pi * 35.0 * t)  # Attenuated freq
    # x += a * np.exp(-20 * (t - 4*T/8)**2) * np.cos(2 * np.pi * 50.0 * t)  # Attenuated freq

    # Square pulse
    for k in range(20, 20 + int(fs1/2)):  # 0.5 sec
        input_signal[k] -= 1

    # Noise
    input_signal += 0.1 * np.cos(2 * np.pi * 30 * t)
    input_signal += 0.1 * np.cos(2 * np.pi * 40 * t)
    input_signal += 0.1 * np.cos(2 * np.pi * 50 * t)

    # Plot input
    plt.figure(2)
    plt.clf()
    plt.plot(t, input_signal, label='Noisy signal')

    # ---------------------------- FILTER INPUT SIGNAL ----------------------------
    # Filter 100Hz signal and plot output
    for order in [1]:  # , 2]:
        y = apply_butter_bandpass(input_signal, lowcut, highcut, fs1, order=order)
        plt.plot(t, y, label='Filtered signal order=%d' % (2*order))

    # Filter 400Hz signal and plot output
    nsamples = T * fs2
    # Redefine t and x for 400 samples/seg
    input_interp = interpolate.interp1d(t, input_signal)
    t_400hz = np.linspace(0, t[len(t) - 1], nsamples, endpoint=False)
    input_signal_400hz = input_interp(t_400hz)
    # y = butter_bandpass_filter(x, lowcut, highcut, fs2, order=1)
    # plt.plot(t, y, 'y.', label='Filtered signal for 400Hz')

    # ------------------------ PLOT FINAL IMPLEMENTATION --------------------------
    # Final implementation for 100Hz
    y = final_implementation_100hz(input_signal)
    plt.plot(t, y, '+k', label='Final implementation for 100Hz')
    # Final implementation for 400Hz
    y_400hz = final_implementation_400hz(input_signal_400hz)
    plt.plot(t_400hz, y_400hz, '.y', label='Final implementation for 400Hz')

    plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()
