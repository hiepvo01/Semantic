from pl_bolts.models.autoencoders import AE

class MyAEFlavor(AE):
    
    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
        encoder = YourSuperFancyEncoder(...)
        return encoder

model = AE()
trainer = Trainer()
trainer.fit(model)