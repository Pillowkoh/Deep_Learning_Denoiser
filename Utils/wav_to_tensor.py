import tqdm
import torch
from os import listdir
from os.path import isfile, join

path_input = '../data/clean_audio_WAV'
path_output = '../data/clean_audio_tensors'

# path_input = './data/noisy_audio_WAV'
# path_output = './data/noisy_audio_tensors'

file_data = [f for f in listdir(path_input) if isfile (join(path_input, f))]
for line in tqdm(file_data):
    if ( line[-1:] == '\n' ):
        line = line[:-1]

    # Reading Song
    songname = path_input + '/' + line
    save_dest = path_output + '/' + line.split('.')[0] + '.pt'
    print(save_dest)

    waveform, sample_rate = torch.load(songname)
    torch.save(waveform, save_dest)
