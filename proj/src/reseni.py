import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as ss
import scipy.fft as fft

tones_arr = ((35, 61.74), (55, 196.00), (94, 1864.66))


#####################################TASK 1###################################################

MIDIFROM = 24
MIDITO = 108
SKIP_SEC = 0.25
HOWMUCH_SEC = 0.5
WHOLETONE_SEC = 2

howmanytones = MIDITO - MIDIFROM + 1
tones = np.arange(MIDIFROM, MIDITO+1)

original, Fs = sf.read('/home/fiskaissad/vutfit/iss/ISSprojekt/klavir.wav')

N = int(Fs * HOWMUCH_SEC) # pocet vzorkov v 0.5 sekundach tonu
Nwholetone = int(Fs * WHOLETONE_SEC) # pocet vzorkov v celom tone (2)

xall = np.zeros((MIDITO+1, N)) # matrix with all tones - first signals empty,
# but we have plenty of memory ...


samplefrom = int(SKIP_SEC * Fs)
sampleto = samplefrom + N
for tone in tones:
    x = original[samplefrom:sampleto]
    x = x - np.mean(x) # safer to center ...
    xall[tone,:] = x
    samplefrom += Nwholetone
    sampleto += Nwholetone
    
whole_tone_A = original[(tones_arr[0][0]-MIDIFROM)*Nwholetone:(tones_arr[0][0]-MIDIFROM+1)*Nwholetone]
whole_tone_B = original[(tones_arr[1][0]-MIDIFROM)*Nwholetone:(tones_arr[1][0]-MIDIFROM+1)*Nwholetone]
whole_tone_C = original[(tones_arr[2][0]-MIDIFROM)*Nwholetone:(tones_arr[2][0]-MIDIFROM+1)*Nwholetone]
    
# 0.5s tones
tone35 = xall[tones_arr[0][0]]
tone55 = xall[tones_arr[1][0]]
tone94 = xall[tones_arr[2][0]]

from scipy.io.wavfile import write
write('a_orig.wav', Fs, tone35)
write('b_orig.wav', Fs, tone55)
write('c_orig.wav', Fs, tone94)


##########################################TONE 35################################################

# #ton 35 midi f = 61.74 hz
# # sf.write('/home/fiskaissad/vutfit/iss/ISSprojekt/audio/a_orig.wav', tone35, Fs//2)
# plt.subplot(211)

# t35period3 = 3*(int(Fs/tones_arr[0][1]))
# perioda35 = xall[35][0:t35period3]

# plt.xlabel("Time [ms]")
# plt.ylabel("Amplitude")
# plt.title("Tone 35")

# plt.tight_layout()
# plt.plot(perioda35)
# plt.subplot(212)

# dft35 = np.fft.rfft(xall[35]) + 10**(-5)

# spectral35 = np.abs(dft35)
# spectral35 = spectral35[:spectral35.size // 2]

# LOGspectral35 = np.log(spectral35)

# plt.xlabel("Frequency [Hz]")
# plt.ylabel("log PSD")
# plt.title("Spektrum 35")

# plt.tight_layout()
# plt.plot(LOGspectral35)
# plt.savefig('Ton35.png')


##########################################TONE 55################################################

# #ton 55 midi f = 196 hz
# # sf.write('/home/fiskaissad/vutfit/iss/ISSprojekt/audio/b_orig.wav', tone55, Fs//2)
# plt.subplot(211)

# t55period3 = 3*(int(Fs/tones_arr[1][1]))
# perioda55 = xall[55][0:t55period3]

# plt.xlabel("Time [ms]")
# plt.ylabel("Amplitude")
# plt.title("Tone 55")
# plt.tight_layout()
# plt.plot(perioda55)

# plt.subplot(212)

# dft55 = np.fft.rfft(xall[55]) + 10**(-5)

# spectral55 = np.abs(dft55)
# spectral55 = spectral55[:spectral55.size // 2]

# LOGspectral55 = np.log(spectral55)

# plt.xlabel("Frequency [Hz]")
# plt.ylabel("log PSD")
# plt.title("Spektrum 55")
# plt.tight_layout()
# plt.plot(LOGspectral55)
# plt.savefig('Ton55.png')


##########################################TONE 94################################################

#ton 94 midi f = 1864.66 hz
# sf.write('/home/fiskaissad/vutfit/iss/ISSprojekt/audio/c_orig.wav', tone94, Fs//2)
plt.subplot(211)

t94period3 = 3*(int(Fs/tones_arr[2][1]))
perioda94 = xall[94][0:t94period3]

plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.title("Tone 94")
plt.tight_layout()
plt.plot(perioda94)

plt.subplot(212)

dft94 = np.fft.rfft(xall[94]) + 10**(-5)

spectral94 = np.abs(dft94)
spectral94 = spectral94[:spectral94.size // 2]

LOGspectral94 = np.log(spectral94)

plt.xlabel("Frequency [Hz]")
plt.ylabel("log PSD")
plt.title("Spektrum 94")
plt.tight_layout()
plt.plot(LOGspectral94)
plt.savefig('Ton94.png')


plt.tight_layout()
plt.plot(xall[24])
plt.savefig('out.png')
plt.show()

###########################################TASK 2###################################################
import scipy.signal as signal

fig, axs = plt.subplots(3)

fund_freq_approx = np.zeros(MIDITO+1)
fund_freq_correlation = np.zeros(MIDITO+1)
fund_freq_dft = np.zeros(MIDITO+1)

for i in range(MIDIFROM, MIDITO+1, 1):
    dtft = np.abs(fft.rfft(xall[i]))
    fund_freq_dft[i] = np.fft.rfftfreq(xall[i].shape[0], 1/Fs)[np.argmax(dtft)]
    

    correlation = signal.correlate(xall[i], xall[i], mode='full')
    correlation = correlation[correlation.size//2:]
    peaks, _ = signal.find_peaks(correlation)  
    fund_freq_correlation[i] = Fs / np.where(correlation == max(correlation[peaks]))[0]
    
    
    if i >= 40:
        fund_freq_approx[i] = fund_freq_dft[i]
    else:
        fund_freq_approx[i] = fund_freq_correlation[i]

    if i==tones_arr[0][0]:
        axs[0].set_title("Tone 35 (correlation)")
        axs[0].set_xlabel("Frequency [Hz]")
        axs[0].set_ylabel("Amplitude")
        
        plt.tight_layout()
        axs[0].plot(np.arange(correlation.shape[0]), correlation)
        axs[0].axvline(np.where(correlation == max(correlation[peaks]))[0], color='lime', linestyle='--')
        

    if i==tones_arr[1][0]:
        axs[1].set_title("Tone 55 (DFT):")
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("log PSD")
        
        plt.tight_layout()
        axs[1].plot(np.fft.rfftfreq(xall[i].shape[0], 1/Fs), np.log(np.abs(dtft)**2+1e-5))
        axs[1].axvline(fund_freq_approx[tones_arr[1][0]], color='lime', linestyle='--')
        

    if i==tones_arr[2][0]:
        axs[2].set_title("Tone 94 (DFT):")
        axs[2].set_xlabel("Frequency [Hz]")
        axs[2].set_ylabel("log PSD")
        
        plt.tight_layout()
        axs[2].plot(np.fft.rfftfreq(xall[i].shape[0], 1/Fs), np.log(np.abs(dtft)**2+1e-5))
        axs[2].axvline(fund_freq_approx[tones_arr[2][0]], color='lime', linestyle='--')
        
    plt.savefig('fundfreq.png')    
    
print("/////////////////////////////////////")
print("Tone 35:")
print("-------------------------------")
print("-MIDI:           61.74")
print("-correlation:    ", fund_freq_correlation[35], "<--")
print("-dft:            ", fund_freq_dft[35])
print()
print("/////////////////////////////////////")
print("Tone 55:")
print("-------------------------------")
print("-MIDI:           196.00")
print("-correlation:    ", fund_freq_correlation[55], "<--")
print("-dft:            ", fund_freq_dft[55])
print()
print("/////////////////////////////////////")
print("Tone 94:")
print("-------------------------------")
print("-MIDI:   1864.66")
print("-correlation:  ", fund_freq_correlation[94])
print("-dft:   ", fund_freq_dft[94], "<--")