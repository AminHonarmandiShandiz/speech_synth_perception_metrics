import subprocess

import soundfile as sf
import torch

import ustool.ustools.voice_activity_detection as vad
import WaveGlow_functions
import ustool.TALTool.visualiser.tools.utils as utils
import librosa
import ustool.TALTool.visualiser.tools.io as myio
import skimage.transform
import h5py as hp
from scipy.io import wavfile
import os
import numpy as np
import argparse

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


def makingDS(inputpath: str, outputpath: str, orgDSPath: str, synthesize: bool, waveglow_path: str, dataset: str):
    inputpath = open(inputpath, 'r')
    c = 0
    t = 0
    frames = 164
    cut = 164
    count = 0

    if synthesize:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        waveglow_name = 'WaveGlow-EN'
        print('loading WaveGlow model...')
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda()

    for f in inputpath:
        day = f.split(' ')[0]
        fileofday = f.split(' ')[1]
        filename = fileofday.split('\n')[0]
        file = fileofday.split('.wav')[0]
        Full_input_path = orgDSPath + '/' + day + '/'
        prefix = file.split('_')[0]
        prefix = int(prefix)
        if prefix < 50:
            continue

        wav, wav_sr = librosa.load(os.path.join(Full_input_path, filename), sr=48000)
        ult, params = myio.read_ultrasound_tuple(os.path.join(Full_input_path, file), shape='3d', cast=None,
                                                 truncate=None)
        vid, meta = myio.read_video(os.path.join(Full_input_path, file), shape='3d', cast=None)
        ult = resize(ult, 128)

        wav = librosa.core.resample(wav, orig_sr=48000, target_sr=22050, res_type='kaiser_best')
        ult, vid, wav = utils.trim_to_parallel_streams(ult, vid, wav, params, meta, samplingFrequency)

        time_segments = vad.detect_voice_activity(wav, 22050)
        silence, speech = vad.separate_silence_and_speech(wav, wav_sr, time_segments)

        speech_wav = "speech.wav"
        wavfile.write(filename=speech_wav, rate=wav_sr, data=speech)

        wav, wav_sr = librosa.load(speech_wav, sr=wav_sr)

        wav = librosa.core.resample(wav, orig_sr=16000, target_sr=22050, res_type='kaiser_best')

        sf.write(speech_wav, wav,
                 22050,
                 subtype='PCM_16')

        hop_length_UTI = int(samplingFrequency / params['FramesPerSec'])
        # wavglow features

        stft = WaveGlow_functions.TacotronSTFT(filter_length=1024, hop_length=hop_length_UTI, \
                                               win_length=1024, n_mel_channels=n_melspec,
                                               sampling_rate=samplingFrequency, \
                                               mel_fmin=0, mel_fmax=8000)

        mel_data = WaveGlow_functions.get_mel(speech_wav, stft)
        os.remove(speech_wav)
        mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))

        x = mel_data.shape[0]
        y = ult.shape[0]
        z = abs(x - y)
        if x != y:
            if x > y:
                mel_data = mel_data[:-z, :]

            else:
                ult = ult[:-z, :, :]

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

        c += 1
        label = np.ones(cut)
        label = label * prefix
        if c == 1:
            X_dev = ult
            y_dev = mel_data
            y_lbl = label

        else:
            X_dev = np.concatenate((X_dev, ult), axis=0)
            y_dev = np.concatenate((y_dev, mel_data), axis=0)
            y_lbl = np.concatenate((y_lbl, label), axis=0)

        if synthesize:

            interpolate_ratio = hop_length_UTI / hop_length_WaveGlow
            melspec_predicted = skimage.transform.resize(mel_data, \
                                                         (int(mel_data.shape[0] * interpolate_ratio),
                                                          mel_data.shape[1]), preserve_range=True)

            mel_data_for_synth = np.rot90(np.fliplr(melspec_predicted), axes=(0, 1))
            mel_data_for_synth = torch.from_numpy(mel_data_for_synth.copy()).float().to(device)
            # reverse

            with torch.no_grad():
                audio = waveglow.infer(mel_data_for_synth.view([1, 80, -1]).cuda(), sigma=0.666)
                audio = audio[0].data.cpu().numpy()
                # mel = np.transpose(mel_data)
                # audio = librosa.feature.inverse.mel_to_audio(mel, sr=22050, n_fft=1024, hop_length=270)
                sf.write(outputpath + '/' + dataset + '/synchOrgwav/' + str(prefix) + '_melOrg.wav', audio,
                         22050, subtype='PCM_16')

    with hp.File(outputpath + '/' + dataset + '/' + 'TestInputFile30.h5', 'w') as h5:

        h5.create_dataset('Xvalues', data=X_dev)

    with hp.File(outputpath + '/' + dataset + '/' + 'TestTargetFile30.h5', 'w') as h5:
        h5.create_dataset('Yvalues', data=y_dev)
    # labeling
    with hp.File(outputpath + '/' + dataset + '/' + 'Testlabel30.h5', 'w') as h5:
        h5.create_dataset('Yvalues', data=y_lbl)


def check_ffmpeg():
    try:
        # Run a simple ffmpeg command to check if it is installed
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("ffmpeg is installed and ready to use.")
            print(result.stdout.split('\n')[0])  # Print the first line of the version info
        else:
            print("ffmpeg is not installed or not functioning properly.")
            print(result.stderr)
    except FileNotFoundError:
        print("ffmpeg is not installed.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating Datasets")
    parser.add_argument("-i", "--input_path", type=str, help="Path to input list file", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="Path to the save output", required=True)
    parser.add_argument("-ds", "--dataset_path", type=str, help="Path to TAL dataset", required=True)
    parser.add_argument("-dst", "--dataset_type", type=str, help="type of the dataset (test, validation, train)", required=False, default='test')
    parser.add_argument("-sz", "--synthesize", type=bool, help="if True then synthesizing the utterance during preparing the dataset using mel data", required=False, default=False)
    parser.add_argument("-wg", "--waveglow_path", type=str, help="waveglow vocoder to synthesize speech from mel", required=False, default='/waveglow_256channels_ljs_v1.pt')
    args = parser.parse_args()
    # check_ffmpeg()
    makingDS(args.input_path, args.output_path, args.dataset_path, args.synthesize, args.waveglow_path, args.dataset_type)
    print('done')


