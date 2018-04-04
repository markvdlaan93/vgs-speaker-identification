import numpy

val_conv = numpy.load('data/flickr8k_val_conv.npy')
val_emb = numpy.load('data/flickr8k_val_emb.npy')
val_rec = numpy.load('data/flickr8k_val_rec.npy')
val_spk = numpy.load('data/flickr8k_val_spk.npy')
val_text = numpy.load('data/flickr8k_val_text.npy')
val_mfcc = numpy.load('data/flickr8k_val_mfcc.npy')

def shapes(datasets):
   for dataset in datasets:
      print(dataset.shape)

shapes([val_conv, val_emb, val_rec, val_spk, val_text, val_mfcc])