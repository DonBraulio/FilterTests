from scipy.signal import butter, lfilter, square
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz


def final_implementation(u):
    y = [0.0]*len(u)
    for k in range(0, len(u)):
        u_2 = u[k-2] if k > 1 else 0
        y_2 = y[k-2] if k > 1 else 0
        y_1 = y[k-1] if k > 0 else 0
        y[k] = 0.4984292*(u[k] - u_2) + 0.9968584 * y_1 - 0.0031416 * y_2
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 100.0
    lowcut = 0.1
    highcut = 25.0

    # f_scale = np.asarray([10 ** (r / 100) for r in range(-300, 200)])
    # w_scale = f_scale / fs  # Normalize by fs

    plt.close('all')
    # Plot the frequency response for a few different orders.
    f, (mag_plot, phase_plot) = plt.subplots(2, sharex=True)
    for order in [1, 2]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        print("Numerator for order %d: " % order, b)
        print("Denominator for order %d: " % order, a)
        w, h = freqz(b, a, worN=5000)
        mag_plot.semilogx((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % (2*order))
        phase_plot.semilogx((fs * 0.5 / np.pi) * w, np.angle(h)*180/np.pi, label="order = %d" % (2*order))

    mag_plot.set_title('Gain')
    phase_plot.set_title('Phase (deg)')
    mag_plot.grid(True)
    phase_plot.grid(True)
    mag_plot.legend(loc='best')
    phase_plot.legend(loc='best')

    plt.xlabel('Frequency (Hz)')
    plt.show()

    # Filter noisy signal
    T = 10.0
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 1
    x = 2  # DC offset
    x += a * np.exp(-20 * (t - 1*T/8)**2) * np.cos(2 * np.pi * 5.0 * t)  # Passband wave
    x += a * np.exp(-20 * (t - 2*T/8)**2) * np.cos(2 * np.pi * 10.0 * t)  # Passband wave
    x += a * np.exp(-20 * (t - 3*T/8)**2) * np.cos(2 * np.pi * 15.0 * t)  # Passband wave
    x += a * np.exp(-20 * (t - 4*T/8)**2) * np.cos(2 * np.pi * 25.0 * t)  # Pole frequency
    # x += a * np.exp(-20 * (t - 3*T/8)**2) * np.cos(2 * np.pi * 35.0 * t)  # Attenuated freq
    # x += a * np.exp(-20 * (t - 4*T/8)**2) * np.cos(2 * np.pi * 50.0 * t)  # Attenuated freq

    # Add pulse of -1 amplitude and 0.5 seg
    for k in range(20, 20 + int(fs/2)):
        x[k] -= 1

    # x += a * square(2 * np.pi * f0 * t)
    x += 0.1 * np.cos(2 * np.pi * 30 * t)
    x += 0.1 * np.cos(2 * np.pi * 40 * t)
    x += 0.1 * np.cos(2 * np.pi * 50 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    for order in [1, 2]:
        y = butter_bandpass_filter(x, lowcut, highcut, fs, order=order)
        plt.plot(t, y, label='Filtered signal order=%d' % (2*order))
    #y = final_implementation(x)
    #plt.plot(t, y, '+k', label='Final implementation')
    plt.xlabel('time (seconds)')
    # plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()