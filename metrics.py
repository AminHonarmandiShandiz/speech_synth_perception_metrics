import argparse
import librosa
import torchmetrics
import torch
from pymcd.mcd import Calculate_MCD
from asteroid_filterbanks import STFTFB, Encoder, transforms
from asteroid.losses import SingleSrcPMSQE


def main(args):
    # Load original audio
    y_original, sr_original = librosa.load(args.original_filepath, sr=16000)

    # Load generated audio
    y_generated, sr_generated = librosa.load(args.generated_filepath, sr=16000)

    # Ensure that both audio files have the same length
    min_length = min(len(y_original), len(y_generated))
    y_original = y_original[:min_length]
    y_generated = y_generated[:min_length]
    # Convert audio data to PyTorch tensors
    y_original_tensor = torch.tensor(y_original)
    y_generated_tensor = torch.tensor(y_generated)

    if args.pesq or args.all:
        # Initialize the pesq metric
        wb_pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(int(sr_original), 'wb')
        nb_pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(int(sr_original), 'nb')
        # Compute PESQ
        wb_pesq = wb_pesq_metric(y_generated_tensor, y_original_tensor)
        nb_pesq = nb_pesq_metric(y_generated_tensor, y_original_tensor)
        print("Wide-band PESQ:", round(wb_pesq.item(), 4))
        print("Narrow-band PESQ:", round(nb_pesq.item(), 4))
    if args.stoi or args.all:
        # Initialize the stoi metric
        stoi_metric = torchmetrics.audio.ShortTimeObjectiveIntelligibility(int(sr_original))
        # Compute STOI
        stoi = stoi_metric(y_generated_tensor, y_original_tensor)
        print("STOI:", round(stoi.item(), 4))
    if args.estoi or args.all:
        estoi_metric = torchmetrics.audio.ShortTimeObjectiveIntelligibility(int(sr_original), extended=True)
        estoi = estoi_metric(y_generated_tensor, y_original_tensor)
        print("ESTOI:", round(estoi.item(), 4))
    if args.sisdr or args.all:
        sisdr_metric = torchmetrics.audio.ScaleInvariantSignalDistortionRatio()
        sisdr = sisdr_metric(y_generated_tensor, y_original_tensor)
        print("SI-SDR Score:", round(sisdr.item(), 4))
    if args.sdr or args.all:
        sdr_metric = torchmetrics.audio.SignalDistortionRatio()
        sdr = sdr_metric(y_generated_tensor, y_original_tensor)
        print("SDR:", round(sdr.item(), 4))
    if args.pmsqe or args.all:

        # Compute magnitude spectra
        stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
        ref_spec = transforms.mag(stft(torch.tensor(y_original).unsqueeze(0)))
        est_spec = transforms.mag(stft(torch.tensor(y_generated).unsqueeze(0)))

        # Compute PMSQE loss
        loss_func = SingleSrcPMSQE()
        loss_value = loss_func(est_spec, ref_spec)
        print("PMSQE: ", round(loss_value.item(), 4))

    if args.mcd or args.all:
        mcd_toolbox = Calculate_MCD(MCD_mode="plain")
        mcd_value = mcd_toolbox.calculate_mcd(args.original_filepath, args.generated_filepath)
        print("MCD: ", round(mcd_value.item(), 4))


# PESQ, STOI, ESTOI, SISDR, SDR, PMSQE, MCD
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Grading Script")
    parser.add_argument("-o", "--original_filepath", type=str, help="Path to the original audio file", required=True)
    parser.add_argument("-g", "--generated_filepath", type=str, help="Path to the generated audio file", required=True)
    parser.add_argument("-a", "--all", type=str, help="Flag to run all available metrics")
    parser.add_argument("-pesq", action="store_true", help="Flag to run PESQ grading")
    parser.add_argument("-stoi", action="store_true", help="Flag to run STOI grading")
    parser.add_argument("-estoi", action="store_true", help="Flag to run ESTOI grading")
    parser.add_argument("-sisdr", action="store_true", help="Flag to run SISDR grading")
    parser.add_argument("-sdr", action="store_true", help="Flag to run SDR grading")
    parser.add_argument("-pmsqe", action="store_true", help="Flag to run PMSQE grading")
    parser.add_argument("-mcd", action="store_true", help="Flag to run MCD grading")
    args = parser.parse_args()

    main(args)
