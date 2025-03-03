import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from models import CodeBigVGAN as Generator
from tqdm import tqdm

h = None
device = None
torch.backends.cudnn.benchmark = False

MAX_WAV_VALUE = 32768.0

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def inference(a, h):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    try:
        input_file = open(a.input_code_file, 'r', encoding='utf-8')
        input_lines = input_file.readlines()
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, trying UTF-16...")
        input_file = open(a.input_code_file, 'r', encoding='utf-16')
        input_lines = input_file.readlines()
    except UnicodeDecodeError:
        print("UTF-16 encoding failed, trying Latin-1 as fallback...")
        input_file = open(a.input_code_file, 'r', encoding='latin-1')
        input_lines = input_file.readlines()
    finally:
        input_file.close()
    
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    
    syn_dir = os.path.join(a.output_dir, 'syn')  
    if not os.path.exists(syn_dir):  
        os.makedirs(syn_dir) 
        
    for i in tqdm(range(len(input_lines))):
        if input_lines[i][0] != 'D':
            continue
        generated_line = input_lines[i]

        item_name,  gen_code = generated_line.split('\t')
        item_name = item_name[2:]

        item_fname = item_name + '.wav'
        gen_fpath = os.path.join(a.output_dir, 'syn', item_fname) 
        
        audio_str = gen_code
        audio = audio_str.split()
        ori_audio = audio

        for i, x in enumerate(ori_audio):
            if (x == '<acoustic_start>' and ori_audio[i + 1] != '<acoustic_start>'):
                audio = audio[i + 1 : -4]
                break
        
        gen_code = [int (x[4:]) for x in audio if len(x) > 0]

        if len(gen_code) % 3 == 1:
            gen_code = gen_code[:-1]
        elif len(gen_code) % 3 == 2:
            gen_code = gen_code[:-2]
        
        for i, x in enumerate(gen_code):
            if gen_code[i] >= 2048:
                gen_code[i] -= 2048
                continue
            if gen_code[i] >= 1024:
                gen_code[i] -= 1024
                
        audio = torch.LongTensor(gen_code).reshape(-1, 3).transpose(0, 1).cuda()
        gen_code = np.array(gen_code)
        gen_code = torch.LongTensor(np.stack([gen_code[::3], gen_code[1::3], gen_code[2::3]])).unsqueeze(0).to(device)

        with torch.no_grad():
            y_g_hat = generator(gen_code)

        audio_g = y_g_hat.squeeze().detach() * MAX_WAV_VALUE
        audio_g = audio_g.cpu().numpy().astype('int16')

        write(gen_fpath, h.sampling_rate, audio_g)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_code_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint_file', required=True)

    a = parser.parse_args()

    config_file = os.path.join(os.path.dirname(a.checkpoint_file), 'config.json')
    
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found.")
        print("Creating a default config.json file...")
        
        default_config = {
            "resblock": "1",
            "num_gpus": 1,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
            "seed": 1234,
            "upsample_rates": [5, 4, 2, 2],
            "upsample_kernel_sizes": [10, 8, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "segment_size": 8192,
            "num_mels": 80,
            "num_freq": 1025,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": 8000,
            "fmax_for_loss": None,
            "num_workers": 4,
            "dist_config": {
                "dist_backend": "nccl",
                "dist_url": "tcp://localhost:54321",
                "world_size": 1
            }
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"Default config created at {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            data = f.read()
    except UnicodeDecodeError:
        print("UTF-8 encoding failed when reading config, trying Latin-1...")
        with open(config_file, 'r', encoding='latin-1') as f:
            data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h)


if __name__ == '__main__':
    main()

