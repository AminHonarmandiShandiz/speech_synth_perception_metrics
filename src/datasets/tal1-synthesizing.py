import pickle
import json
import numpy as np

np.random.seed(42)
import ast
import random

random.seed(42)
from ustool.ustools.visualise_ultrasound import display_2d_ultrasound_frame
from ustool.ustools.transform_ultrasound import transform_ultrasound
import webrtcvad
import pickle
import soundfile as sf
import ustool.ustools.voice_activity_detection as vad
import WaveGlow_functions
import ustool.TALTool.visualiser.tools.utils as utils
import librosa
import ustool.TALTool.visualiser.tools.io as myio
import pandas as pds
import skimage.transform
import torch
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer, Dropout, BatchNormalization, Reshape, Bidirectional, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, Adadelta, SGD
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from keras.layers import Dense, Flatten, InputLayer, Dropout, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, \
    BatchNormalization
import h5py as hp
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import os
# from keras.backend import sigmoid
from keras.activations import sigmoid

# from keras.utils.generic_utils import get_custom_objects
from keras.saving import get_custom_objects

from keras.layers import Activation
from scipy.io import wavfile
import os
import numpy as np
import argparse
from tqdm import tqdm
import scipy
# import cv2
import PIL

sts = 5
window_size = sts * 4 + 1
# waveglow parameters
hop_length_UTI = 270  # 12 ms
hop_length_WaveGlow = 256
samplingFrequency = 22050
n_melspec = 80


def resize(data, target_frames):
    ''' resize data stream '''
    z = data.shape[2]
    x, y = data.shape[0], data.shape[1]
    output_shape = (x * y, target_frames)

    data = data.reshape(-1, z)
    resized = skimage.transform.resize(data, output_shape=output_shape, order=1, mode='edge', clip=True,
                                       preserve_range=True, anti_aliasing=True)

    resized = resized.reshape(x, y, -1)

    return resized


# def makingspectrum(inp, outp):
#     with files in os.listdir(inp):
#         with hp.File(inp + files, 'r') as h5:
#             data = h5.get('Xvalues')
#             data = np.array(data)
#
#         mel_data = model.predict(data)
#         n_to_skip = np.floor(window_size // 2).astype(np.int64)
#         # mel_data=mel_data[:,:-1]
#         mel_data = mel_data[n_to_skip:(mel_data.shape[0] - n_to_skip)]
#         # mel_data=
#


def makingDS(inputpath, outputpath, orgDSPath):
    # orgDSPath = r'C:\cygwin64\home\Kontoli\TaL1\core'
    dataset = 'test'
    c = 0
    t = 0
    frames = 164
    cut = 164
    count = 0
    speaker = 0
    mean_std = {}
    flag = False
    spec = []
    for f in inputpath:

        print(f)
        day = f.split(' ')[0]

        fileofday = f.split(' ')[1]
        filename = fileofday.split('\n')[0]
        file = fileofday.split('.wav')[0]
        Full_input_path = orgDSPath + '/' + day + '/'
        prefix = file.split('_')[0]
        # prefix = day[0:2] # wingman
        prefix = int(prefix)
        # if c>= prefix*12:
        #     continue
        if prefix < 50:
            continue

        # if day == 'day3' or day == 'day2' or day == 'day4' or day == 'day5':
        #     continue
        # x,y=scipy.io.wavfile.read('E:/university/UniSzegedProjects/AllSemesters/3-semester/synthesise/edinburg/codes/samples/to_be_eval/ref/08me-016_xaud-ref.wav')
        wav, wav_sr = librosa.load(os.path.join(Full_input_path, filename), sr=48000)
        ult, params = myio.read_ultrasound_tuple(os.path.join(Full_input_path, file), shape='3d', cast=None,
                                                 truncate=None)
        vid, meta = myio.read_video(os.path.join(Full_input_path, file), shape='3d', cast=None)
        ult = resize(ult, 128)

        wav = librosa.core.resample(wav, orig_sr=48000, target_sr=22050, res_type='kaiser_best')
        ult, vid, wav = utils.trim_to_parallel_streams(ult, vid, wav, params, meta, samplingFrequency)
        # wav, wav_sr = librosa.load(outputpath + dataset + day + '/synchOrgwav/' + prefix + '_melOrg.wav', sr=22050)

        # time_segments, wav, wav_sr = vad.detect_voice_activity(wav, 22050) # wingman off
        time_segments = vad.detect_voice_activity(wav, 22050)
        # vad.visualise_voice_activity_detection(wav,wav_sr,time_segments)
        silence, speech = vad.separate_silence_and_speech(wav, wav_sr, time_segments)

        speech_wav = "speech.wav"
        # # new_speech_wav = "new_speech.wav"
        wavfile.write(filename=speech_wav, rate=wav_sr, data=speech)
        # # wavfile.write(filename=silence_wav,rate=wav_sr,data=silence)
        # # os.remove(speech_wav)
        # # os.remove(silence_wav)
        #
        #
        #
        wav, wav_sr = librosa.load(speech_wav, sr=wav_sr)
        #
        wav = librosa.core.resample(wav, orig_sr=16000, target_sr=22050, res_type='kaiser_best')
        #
        sf.write(speech_wav, wav,
                 22050,
                 subtype='PCM_16')

        hop_length_UTI = int(samplingFrequency / params['FramesPerSec'])
        # wavglow features

        stft = WaveGlow_functions.TacotronSTFT(filter_length=1024, hop_length=hop_length_UTI, \
                                               win_length=1024, n_mel_channels=n_melspec,
                                               sampling_rate=samplingFrequency, \
                                               mel_fmin=0, mel_fmax=8000)
        # sf.write(outputpath+ dataset+day + '/resampled&synchOrgwav/' + prefix+ '_orgResampled.wav', wav, 22050,
        #          subtype='PCM_16')
        mel_data = WaveGlow_functions.get_mel(speech_wav, stft)
        os.remove(speech_wav)
        mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))
        # labeling
        # label=vad.silence_labelingsignal(wav,ult)

        # ------
        # mel_data = librosa.feature.melspectrogram(wav, sr=22050, n_fft=1024, hop_length=hop_length_UTI)
        # mel_data = np.transpose(mel_data)
        # os.remove(speech_wav)
        x = mel_data.shape[0]
        y = ult.shape[0]
        z = abs(x - y)
        if x != y:
            if x > y:
                mel_data = mel_data[:-z, :]

            else:
                ult = ult[:-z, :, :]
                # label = label[:-z]
        tmp = len(ult)
        if tmp <= frames:
            t += 1
            continue
        strt = tmp - frames
        rd = np.random.randint(0, strt)
        ult = ult[rd:rd + cut]
        mel_data = mel_data[rd:rd + cut]

        ult = np.array(ult)

        ult = ult.astype('float32')
        count += 1

        for i in range(ult.shape[0]):
            ult[i] = (ult[i] - (np.min(ult[i]))) / (np.max(ult[i]) - np.min(ult[i]))
        # ult/=255.0
        # for i in range(10):
        #     display_2d_ultrasound_frame(ult[i*2],dpi=None, figsize=(10, 10)   )     # mean = np.expand_dims(mean, axis=1)
        # std = np.expand_dims(std, axis=1)
        c += 1
        label = np.ones(cut)
        label = label * prefix
        if c == 1:

            X_dev = ult
            # y_dev = mel_data
            y_dev = mel_data
            y_lbl = label

        else:

            X_dev = np.concatenate((X_dev, ult), axis=0)
            y_dev = np.concatenate((y_dev, mel_data), axis=0)
            y_lbl = np.concatenate((y_lbl, label), axis=0)
    print('less than 5 seconds:', t)
    print('all the utterances:', c)

    #     with hp.File(outputpath + dataset + day + '/ult/' + prefix + '_ult.h5', 'w') as h5:
    #         h5.create_dataset('Xvalues', data=ult)
    # #
    # #     # for conv3d--windowing-------------
    # #
    #     with hp.File(outputpath + dataset + day + '/mel/' + prefix + '_mel.h5', 'w') as h5:
    #         h5.create_dataset('Yvalues', data=mel_data)
    #
    #     #labeling
    #
    #
    # n_to_skip = np.floor(window_size // 2).astype(np.int64)
    #     # n_to_skip+=2
    #     # # mel_data=mel_data[:,:-1]
    #     mel_data = mel_data[n_to_skip:-n_to_skip]
    #     # wav, wav_sr = librosa.load(outputpath + dataset + day + '/org/' + prefix + '_Org.wav', sr=22050)
    #     # mel_data=mel_data[:n_to_skip]
    #     # # # --------------
    #     interpolate_ratio = hop_length_UTI / hop_length_WaveGlow
    #     melspec_predicted = skimage.transform.resize(mel_data, \
    #                                                  (int(mel_data.shape[0] * interpolate_ratio),
    #                                                   mel_data.shape[1]), preserve_range=True)
    #     #
    #     mel_data_for_synth = np.rot90(np.fliplr(melspec_predicted), axes=(0, 1))
    #     mel_data_for_synth = torch.from_numpy(mel_data_for_synth.copy()).float().to(device)
    #     #reverse
    #
    #
    #     # audio = librosa.feature.inverse.mel_to_audio(mel, sr=22050, n_fft=1024, hop_length=hop_length_UTI)
    #     with torch.no_grad():
    #         audio = waveglow.infer(mel_data_for_synth.view([1, 80, -1]).cuda(), sigma=0.666)
    #         audio = audio[0].data.cpu().numpy()
    #         # endd=n_to_skip*270
    #         #     # q=abs(len(wav)-(q))
    #
    #
    #
    #     # mel = np.transpose(mel_data)
    #     # audio = librosa.feature.inverse.mel_to_audio(mel, sr=22050, n_fft=1024, hop_length=270)
    #         x = len(audio)
    #         # wav=wav[endd:-endd]
    #         y=len(wav)
    #         if x<y:
    #             wav = wav[:x]
    #         else:
    #             audio = audio[:y]
    #         sf.write(outputpath + dataset + day + '/synchOrgwav/' + prefix + '_melOrg.wav', audio,
    #              22050, subtype='PCM_16')
    #         sf.write(outputpath + dataset + day + '/org/' + prefix + '_Org.wav', wav, 22050, subtype='PCM_16')
    #     #
    #     #
    #     print("end", flush=True)

    with hp.File(outputpath + dataset + '/' + 'TestInputFile30.h5', 'w') as h5:

        h5.create_dataset('Xvalues', data=X_dev)

    with hp.File(outputpath + dataset + '/' + 'TestTargetFile30.h5', 'w') as h5:
        h5.create_dataset('Yvalues', data=y_dev)
    # labeling
    with hp.File(outputpath + dataset + '/' + 'Testlabel30.h5', 'w') as h5:
        h5.create_dataset('Yvalues', data=y_lbl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio Grading Script")
    inputpath = open(r'C:\Users\Kontoli\Documents\GitHub\speech_synth_perception_metrics\src\partitions\test_tal1.list', 'r')
    outputpath = r'C:\Users\Kontoli\Documents\GitHub\speech_synth_perception_metrics\src\generateData\tal1/'
    orgDSPath = r'C:\cygwin64\home\Kontoli\TaL1\core'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # waveglow_name = 'WaveGlow-EN'
    # waveglow_path = 'D:/projects/datasets/edinburguniversity/code&files/waveglow_256channels_ljs_v1.pt'
    # print('loading WaveGlow model...')
    # waveglow = torch.load(waveglow_path)['model']
    # melspec_scaler = pickle.load(open('D:/projects/datasets/edinburguniversity/code&files/Alexa_WaveGlow.norm.sav', 'rb'))
    # waveglow.cuda()
    makingDS(inputpath, outputpath)
    print('done')
    # from playsound import playsound
    # playsound('alarm.mp3')
