from tqdm import tqdm
from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

path_input = '../data/clean_audio'
path_output = '../data/clean_audio_WAV'

file_data = [f for f in listdir(path_input) if isfile (join(path_input, f))]
for line in tqdm(file_data):
    if ( line[-1:] == '\n' ):
        line = line[:-1]

    # Reading Song
    songname = path_input + '/' + line
    save_dest = path_output + '/' + line.split('.')[0] + '.wav'
    print(save_dest)

    sound = AudioSegment.from_mp3(songname)
    sound.export(save_dest, format="wav")