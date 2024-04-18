import numpy as np
import os
import torch

import librosa
import sklearn.preprocessing

import torch
import nibabel as nib
import scipy.stats as sps

def get_audio_encoding(sr=44100, n_mfcc=15):
    x, sr = librosa.load(os.path.join(os.getcwd(), "src", "Film stimulus.mp3"), sr=44100)
    x = librosa.resample(y=x, orig_sr=44100, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=x, n_mfcc=n_mfcc)
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    return mfcc.T

class Sub:

    """Responsible for the subject and contains information about his data."""

    subs_with_fmri = ['04', '07', '08', '09', '11', '13', '14', '15', '16', '18',
                      '22', '24', '27', '28', '29', '31', '35', '41', '43', '44',
                      '45', '46', '47', '51', '52', '53', '55', '56', '60', '62']

    def __init__(self, number):
        if not number in Sub.subs_with_fmri:
            raise ValueError(f"У {number} испытуемого отсутствуют снимки фМРТ")
        else:
            self.number = number
        self.path = os.path.join(os.getcwd(), "ds003688", f"sub-{self.number}",
                                 "ses-mri3t", "func", f"sub-{self.number}_ses-mri3t_task-film_run-1_bold.nii.gz")
        self.scan = nib.load(self.path)
        self.data = self.scan.get_fdata()
        self.tensor = torch.permute(torch.tensor(self.data), (3, 0, 1, 2))
        
        background = (self.tensor < 750)
        self.normalized = (self.tensor - self.tensor[~background].mean()) / self.tensor[~background].std()
        self.normalized[self.normalized < sps.norm.ppf(0.01)] = self.normalized.min()
        
        self.normalized = self.normalized.numpy()