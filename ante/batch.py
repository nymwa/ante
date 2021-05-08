class Batch:
    def __init__(self, ei, di = None, do = None, el = None, dl = None):
        self.encoder_inputs = ei
        self.decoder_inputs = di
        self.decoder_outputs = do
        self.encoder_lengths = el
        self.decoder_lengths = dl

    def __len__(self):
        return self.encoder_inputs.shape[1]

    def cuda(self):
        self.encoder_inputs = self.encoder_inputs.cuda()

        if self.decoder_inputs is not None:
            self.decoder_inputs = self.decoder_inputs.cuda()

        if self.decoder_outputs is not None:
            self.decoder_outputs = self.decoder_outputs.cuda()

        return self

