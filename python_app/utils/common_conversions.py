import numpy as np
import ctypes

LINE_IMPEDANCE = 50


def add_dB(dB1, dB2):
    return 10 * np.log10(10 ** (dB1 / 10) + 10 ** (dB2 / 10))


def dBm_to_W(dBm):
    return 0.001 * 10 ** (dBm / 10)


def W_to_dBm(W):
    return 10 * np.log10(W / 0.0001)


def spectrum_to_mV(noise_level, delta_f):
    """
    given the average spectral noise, compute the noise we would see in a direct signal
    """
    power = 10 ** (noise_level / 10) * 0.001
    signal = np.sqrt(power * delta_f * LINE_IMPEDANCE)
    return signal


def temperature_to_spectral_noise(temperature):
    """
    Calculated the thermal noise expected at this temperature in dB/Hz
    """
    return 10 * np.log10(4 * 1.38 * 10 ** (-23) * temperature / 0.0001)


def spectral_noise_to_temperature(spectral_noise):
    """
    Given a unifrom spectral noise in dBm, find the temperature causing it
    """
    return dBm_to_W(spectral_noise) / (4 * 1.38 * 10 ** (-23))


def mV_to_spectral_noise(noise_mV, delta_f):
    """
    Given a fluctuation in mV, evaluate the corresponding spectral noise in dB
    """
    return W_to_dBm(((noise_mV / 1000) ** 2) / delta_f / LINE_IMPEDANCE)


def mV_to_temperature(noise_mW, delta_f):
    """
    Given a fluctuation in mV, evaluate the corresponding temperature causing it
    """
    return spectral_noise_to_temp(mV_to_spectral_noise(noise_mW, delta_f))


def dBm_signal_to_V(signal_dBm):
    """
    given a signal of a certain power, find the mean square voltage that it would case in the line
    We use the fact that
        P = int[ V(t)^2/R dt]
          = V^2/2R
        V = sqrt(2*R*P)
        R = 50 Ohms
    """
    return np.sqrt(2 * dBm_to_W(signal_dBm) * LINE_IMPEDANCE)


def dBm_squareSignal_to_V(signal_dBm):
    """
    conver the power of a square wave to its voltage representation
    """
    return np.sqrt(dBm_to_W(signal_dBm) * LINE_IMPEDANCE)
