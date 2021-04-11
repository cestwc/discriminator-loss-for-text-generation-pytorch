# discriminator-loss-for-text-generation-pytorch
A FastText model discriminator

It could be added as a loss function, side by side with ```torch.nn. NLLLoss```. When a pretrained discriminator model is use, it is supposed to account for a smaller portion of loss, otherwise there may be vanishing gradients.
