import os
import argparse
import hashlib
from datasets import load_dataset
import soundfile as sf
import librosa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='HuggingFace dataset name (e.g., "username/dataset")')
    parser.add_argument('--subset', type=str, default=None, help='Dataset subset/config name')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--column', type=str, default='audio', help='Audio column name')
    parser.add_argument('--output-dir', type=str, default='./audios', help='Output directory')
    parser.add_argument('--num-proc', type=int, default=16, help='Number of parallel processes')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = os.path.abspath(args.output_dir)
    column = args.column

    print(f'Loading dataset: {args.dataset}')
    if args.subset:
        ds = load_dataset(args.dataset, args.subset, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    def export_audio(sample):
        audio = sample[column]
        wav = audio['array']
        sr = audio['sampling_rate']
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        audio_hash = hashlib.md5(wav.tobytes()).hexdigest()
        output_path = os.path.join(output_dir, f'{audio_hash}.wav')
        sf.write(output_path, wav, 16000)
        return sample

    print(f'Exporting {len(ds)} audio files to {output_dir} with {args.num_proc} workers')
    ds.map(export_audio, num_proc=args.num_proc, desc='Exporting')

    print(f'Done. Exported {len(ds)} files.')

if __name__ == '__main__':
    main()
