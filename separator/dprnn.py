    

from speechbrain.pretrained import Pretrained  
import torchaudio
import torch
import torch.nn.functional as F

    
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
    
    def separate_wav(self , wav ):
        
        batch = torch.tensor(wav).to(self.device)
        est_sources = self.separate_batch(batch)
        est_sources = est_sources / est_sources.max(dim=1, keepdim=True)[0]
                
        return [est_sources[:, :, 0].detach().squeeze().cpu().numpy() , est_sources[:, :, 1].detach().squeeze().cpu().numpy()  ]
    
