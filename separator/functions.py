import torchaudio

from .dprnn  import DprnnModel


# path_list:  a list of wav path 
# return_type: return numpy array or tensor

def read_wav( path_list  , return_type = "np"):
        
    assert return_type in ["np",'tensor']
    
    wav_list = []
    
    for path in path_list:
          
        batch ,_  = torchaudio.sox_effects.apply_effects_file( path , effects=[['rate',"8000"]]  )
        wav_list.append(batch)
        
    if return_type == "tensor":
        return batch
    
    return [ b.numpy() for b in  wav_list]



# model_path : the folder contain model
# device:  model device

def load_model( model_path , device  ):
    
    # load model
    run_opts = {"device":device}
    model = DprnnModel.from_hparams(  source=model_path,  savedir=model_path,hparams_file = 'dprnn.yaml' ,run_opts = run_opts)
    
    return model
    

