'''
following implementation uses SVF model
'''

import numpy as np
import numpy as np
from bokeh.plotting import figure, show
# from bokeh.io import output_notebook
from bokeh.palettes import Colorblind
import pydub
import os


class GEcore():
    
    def __init__(self):
        self.effectname = ''
        self.audiofilename = ''
        self.framerate = []
        self.signal = []
        self.read_audiofile()

    def read_audiofile(self):
        print('----------------------')
        name = input('Enter the audio filename you want to read including the extension: ')
        print('----------------------')

        
        filename, file_ext = os.path.splitext(name)
        filename = os.getcwd() + '/guitar-effects/audiofiles/' + name
        self.audiofilename = filename
        audiofile = pydub.AudioSegment.from_file(filename, file_ext)
        audiofile = audiofile.fade_out(2000)
        self.framerate = audiofile.frame_rate
        songdata = []  # Empty list for holding audio data
        channels = []  # Empty list to hold data from separate channels
        songdata = np.frombuffer(audiofile._data, np.int16)
        for chn in range(audiofile.channels):
            channels.append(songdata[chn::audiofile.channels])  # separate signal from channels
        self.signal = np.sum(channels, axis=0) / len(channels)  # Averaging signal over all channels
        self.signal = self.norm_signal(self.signal)  # normalize signal amplitude
        self.plot_signal([self.signal], True)

    def norm_signal(self, input_signal):
        output_signal = input_signal / np.max(np.absolute(input_signal))
        return output_signal
        
    def plot_signal(self, audio_signal, pflag):
        if pflag:
            p = figure(width=900, height=500, title='Audio Signal', 
                       x_axis_label='Time (s)', y_axis_label='Amplitude (arb. units)')
            time = np.linspace(0, np.shape(audio_signal)[1] / self.framerate, np.shape(audio_signal)[1])
            m = int(np.shape(audio_signal)[1] / 2000)
            for n in range(np.shape(audio_signal)[0]):
                labels = 'signal ' + str(n + 1)
                p.line(time[0::m], audio_signal[n][0::m], line_color=Colorblind[8][n], 
                       alpha=0.6, legend_label=labels)
            show(p)
        else:
            pass

    def wahwah(self, input_signal, pflag):
        print('----------------------')
        damp = float(input('Enter the wahwah damping factor (< 0.5): '))
        minf = float(input('Enter minimum center cutoff frequency (~ 500Hz): '))
        maxf = float(input('Enter the maximum center cutoff frequency (~ 5000Hz): '))
        wahf = float(input('Enter the "wah" frequency (~ 2000Hz): '))
        print('----------------------')
        output_signal = np.zeros(len(input_signal))
        outh = np.zeros(len(input_signal))
        outl = np.zeros(len(input_signal))
        delta = wahf / self.framerate
        centerf = np.concatenate((np.arange(minf, maxf, delta), np.arange(maxf, minf, -delta)))
        while len(centerf) < len(input_signal):
            centerf = np.concatenate((centerf, centerf))
        centerf = centerf[:len(input_signal)]

        # NOTE 完整的implementation這邊會根據pedal position變
        f1 = 2 * np.sin(np.pi * centerf[0] / self.framerate)
        outh[0] = input_signal[0]
        output_signal[0] = f1 * outh[0]
        outl[0] = f1 * output_signal[0]
        for n in range(1, len(input_signal)):
            outh[n] = input_signal[n] - outl[n-1] -  2 * damp * output_signal[n-1]
            output_signal[n] = f1 * outh[n] + output_signal[n-1]
            outl[n] = f1 * output_signal[n] + outl[n-1]
            f1 = 2 * np.sin(np.pi * centerf[n] / self.framerate)
        output_signal = self.norm_signal(output_signal)
        # self.plot_signal([input_signal, output_signal], pflag)
        return output_signal, centerf
    

if __name__ == "__main__":

    audio_path = "/Users/michael/Desktop/guitar data/EGDB subset/one sample dry/233.wav"
    base_name = os.path.basename(audio_path)[:-4]

    core = GEcore()
    core.wahwah()

