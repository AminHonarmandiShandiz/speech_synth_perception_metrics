import argparse
import librosa
import torchmetrics
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


def main(args):
    # Load original audio
    y_original, sr_original = librosa.load(args.original_filepath)

    # Load generated audio
    y_generated, sr_generated = librosa.load(args.generated_filepath)

    # Ensure that both audio files have the same length
    min_length = min(len(y_original), len(y_generated))
    y_original = y_original[:min_length]
    y_generated = y_generated[:min_length]
    # Convert audio data to PyTorch tensors
    y_original_tensor = torch.tensor(y_original)
    y_generated_tensor = torch.tensor(y_generated)

    # Perform PESQ
    if args.qesq:
        # Initialize the pesq metric
        wb_pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(int(sr_original), 'wb')
        nb_pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(int(sr_original), 'nb')
        # Compute PESQ
        wb_pesq = wb_pesq_metric(y_generated_tensor, y_original_tensor)
        nb_pesq = nb_pesq_metric(y_generated_tensor, y_original_tensor)
        print("Wide-band PESQ:", wb_pesq)
        print("Narrow-band PESQ:", nb_pesq)
    if args.stoi:
        # Initialize the stoi metric
        stoi_metric = torchmetrics.audio.ShortTimeObjectiveIntelligibility(int(sr_original))
        # Compute STOI
        stoi = stoi_metric(y_generated_tensor, y_original_tensor)
        print("STOI:", stoi)
    if args.estoi:
        estoi_metric = torchmetrics.audio.ShortTimeObjectiveIntelligibility(int(sr_original), extended=True)
        estoi = estoi_metric(y_generated_tensor, y_original_tensor)
        print("ESTOI:", estoi)
    if args.sisdr:
        sisdr_metric = torchmetrics.audio.ScaleInvariantSignalDistortionRatio()
        sisdr = sisdr_metric(y_generated_tensor, y_original_tensor)
        print(f"SI-SDR Score: {sisdr}")
    if args.sdr:
        sdr_metric = torchmetrics.audio.SignalDistortionRatio()
        sdr = sdr_metric(y_generated_tensor, y_original_tensor)
        print("SDR:", sdr)
    if args.pmsqe:
        ...
    if args.mcd:
        ...

# PESQ, STOI, ESTOI, SISDR, SDR, PMSQE, MCD
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Grading Script")
    parser.add_argument("-o", "--original_filepath", type=str, help="Path to the original audio file", required=True)
    parser.add_argument("-g", "--generated_filepath", type=str, help="Path to the generated audio file", required=True)
    parser.add_argument("-pesq", action="store_true", help="Flag to run PESQ grading")
    parser.add_argument("-stoi", action="store_true", help="Flag to run STOI grading")
    parser.add_argument("-estoi", action="store_true", help="Flag to run ESTOI grading")
    parser.add_argument("-sisdr", action="store_true", help="Flag to run SISDR grading")
    parser.add_argument("-sdr", action="store_true", help="Flag to run SDR grading")
    parser.add_argument("-pmsqe", action="store_true", help="Flag to run PMSQE grading")
    parser.add_argument("-mcd", action="store_true", help="Flag to run MCD grading")
    args = parser.parse_args()

    main(args)
