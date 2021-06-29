from vanilla_vae import VanillaVAE
import torch as th



class DVFGenerator():

    def __init__(self, modelPath="Model/vae_model.pt") -> None:
        self.vae = VanillaVAE().cuda()
        self.vae.load_state_dict(th.load(modelPath))

    def sample(self,num,device):
        return self.vae.sample(num,device)