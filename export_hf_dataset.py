import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='HuggingFace dataset name (e.g., "username/dataset")')
    parser.add_argument('--subset', type=str, default=None, help='Dataset subset/config name')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--column', type=str, default='audio', help='Audio column name')
    parser.add_argument('--output-dir', type=str, default='./audios', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading dataset: {args.dataset}')
    if args.subset:
        ds = load_dataset(args.dataset, args.subset, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    print(f'Exporting {len(ds)} audio files to {args.output_dir}')

    for i, sample in enumerate(tqdm(ds)):
        audio = sample[args.column]
        wav = audio['array']
        sr = audio['sampling_rate']

        output_path = os.path.join(args.output_dir, f'{i:08d}.wav')
        sf.write(output_path, wav, sr)

    print(f'Done. Exported {len(ds)} files.')

if __name__ == '__main__':
    main()
