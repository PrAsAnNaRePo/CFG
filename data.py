import os
import argparse
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import spacy
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datetime import timedelta
import numpy as np

spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(str(text))]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class LoadData(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.df = pd.read_csv(file_path)[0:2]
        self.vocab = Vocabulary(freq_threshold=2)
        self.Lables = self.df['name']
        self.vocab.build_vocabulary(self.Lables.tolist()) 
        self.vidLink = self.df['contentUrl'].values
        self.prepare_dataset(self.vidLink)
        dirs = []
        for i in os.listdir('./'):
            if os.path.isdir(f'./{i}') and i != 'models':
                dirs.append(i)
        self.order_frame(dirs)

    def prepare_dataset(self, links):
        print('Downloading videos')
        for i in links:
            os.system(f'wget {i}')
            vid_id = self.df.loc[self.df['contentUrl']==i, 'videoid'].values[0]
            os.system(f'mkdir {vid_id}')
            self.split_frames(i.split('/')[-1], vid_id)

    def format_timedelta(self, td):
        """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
        omitting microseconds and retaining milliseconds"""
        result = str(td)
        try:
            result, ms = result.split(".")
        except ValueError:
            return (result + ".00").replace(":", "-")
        ms = int(ms)
        ms = round(ms / 1e4)
        return f"{result}.{ms:02}".replace(":", "-")


    def get_saving_frames_durations(self, cap, saving_fps):
        """A function that returns the list of durations where to save the frames"""
        s = []
        # get the clip duration by dividing number of frames by the number of frames per second
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        # use np.arange() to make floating-point steps
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        return s


    def split_frames(self, video_file, folder, SAVING_FRAMES_PER_SECOND=10):
        filename, _ = os.path.splitext(video_file)
        # read the video file    
        cap = cv2.VideoCapture(video_file)
        # get the FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # get the list of duration spots to save
        saving_frames_durations = self.get_saving_frames_durations(cap, saving_frames_per_second)
        # start the loop
        count = 0
        while True:
            is_read, frame = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            # get the duration by dividing the frame count by the FPS
            frame_duration = count / fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, 
                # then save the frame
                frame_duration_formatted = self.format_timedelta(timedelta(seconds=frame_duration))
                cv2.imwrite(f"./{folder}/frame{frame_duration_formatted}.jpg", frame) 
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            count += 1

    def order_frame(self, directory):
        self.lable_frames_pairs = []
        for i in directory:
            for j in range(len(os.listdir(i))-1):
                frame1 = cv2.resize(np.array(Image.open(f'{i}/{os.listdir(i)[j]}').convert('RGB')) / 255.0, (256, 256))
                frame2 = cv2.resize(np.array(Image.open(f'{i}/{os.listdir(i)[j+1]}').convert('RGB')) / 255.0, (256, 256))
                numericalized_caption = [self.vocab.stoi["<SOS>"]]
                print(self.df.loc[self.df['videoid']==int(i), 'name'].values[0])
                numericalized_caption = self.vocab.numericalize(self.df.loc[self.df['videoid']==int(i), 'name'].values[0])
                numericalized_caption.append(self.vocab.stoi["<EOS>"])
                self.lable_frames_pairs.append([frame1, numericalized_caption, frame2])

    def __getitem__(self, index):
        frame1 = torch.tensor(self.lable_frames_pairs[index][0], dtype=torch.float32)
        lbl = torch.tensor(self.lable_frames_pairs[index][1], dtype=torch.long)
        frame2 = torch.tensor(self.lable_frames_pairs[index][2], dtype=torch.float32)
        return frame1, lbl, frame2

data = LoadData(r'E:\codes\CFG\results_2M_val.csv')
print(data[0][0].shape, data[0][1])
