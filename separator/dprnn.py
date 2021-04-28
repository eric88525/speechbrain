    

from speechbrain.pretrained import Pretrained  
import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
from numpy import dot
from numpy.linalg import norm


def concate_wav(wav_pair , seg_len , overlap ,rate=8000 ):
    
    if len(wav_pair) == 1:
        return wav_pair
    
    overlap = int(overlap * rate)  
    
    result_wav =  [ wav_pair[0] ]
    
    for i in range( 1 , len(wav_pair) ):
        item = wav_pair[i]
        last_item = wav_pair[i-1]
        if cosine_similarity( last_item[0][-overlap:] ,  item[0][:overlap]  ) < \
           cosine_similarity( last_item[0][-overlap:] ,  item[1][:overlap]  ) :
                
            wav_pair[i][0], wav_pair[i][1] = wav_pair[i][1], wav_pair[i][0]
            
        result_wav.append(  [wav_pair[i][0][overlap:] ,  wav_pair[i][1][overlap:] ]  )
    
    return np.hstack( result_wav )
        
def cosine_similarity (a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))    
    return cos_sim
        
def split_wav( wav , seg_len = 210 , overlap=30 ,rate=8000 ):
             
    seg_len = int(seg_len * rate)
    overlap = int(overlap * rate)
        
    if wav.shape[-1] <= seg_len:
        return [wav]
        
    splited_wav = []
        
    for i in range(  0 ,  wav.shape[-1] - (wav.shape[-1] % (seg_len-overlap) ) + 1,  seg_len-overlap   ):
        splited_wav.append( wav[:,i:i+seg_len]   )
            
    return  splited_wav
  

class DprnnModel(Pretrained):

    MODULES_NEEDED = ["encoder", "masknet", "decoder"]

    def separate_batch(self, mix):
        """Run source separation on batch of audio.

        Arguments
        ---------
        mix : torch.tensor
            The mixture of sources.

        Returns
        -------
        tensor
            Separated sources
        """

        # Separation
        mix_w = self.modules.encoder(mix)
        est_mask = self.modules.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.modules.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        return est_source

    def separate_file(self, path, savedir="."):        
        
        """Separate sources from file.

        Arguments
        ---------
        path : str
            Path to file which has a mixture of sources. It can be a local
            path, a web url, or a huggingface repo.
        savedir : path
            Path where to store the wav signals (when downloaded from the web).
        Returns
        -------
        tensor
            Separated sources
        """
        source, fl = split_path(path)

        # sample rate limit to 8000
        batch ,_  = torchaudio.sox_effects.apply_effects_file( path , effects=[['rate',"8000"]]  )

        # eric fix
        batch = batch.to(self.device)
        
        #print(batch.device)
        
        est_sources = self.separate_batch(batch)
        est_sources = est_sources / est_sources.max(dim=1, keepdim=True)[0]
        return est_sources
    
    def separate_wav(self , wav  , seg_len = 210 , overlap=30 ):
        
        splited_wav = split_wav( wav ,seg_len , overlap )
        temp_result = []
        
        for wav in splited_wav:
            batch = torch.tensor(wav).to(self.device)
            est_sources = self.separate_batch(batch)
            est_sources = est_sources / est_sources.max(dim=1, keepdim=True)[0]     
            est_sources = est_sources.detach().cpu()   
        
            temp_result.append(  [ est_sources[:, :, 0].squeeze().numpy() , 
                                   est_sources[:, :, 1].squeeze().numpy() ]    )
        
        print(f"temp {len(temp_result)}")
        sep_result = concate_wav( temp_result ,seg_len , overlap )
        
        return sep_result
        
        
        

        
    
