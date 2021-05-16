from model import *
from data import *

testGene = testGenerator("data/membrane/test")
model = unet()
model.load_weights("unet_membrane.hdf5")
results = model.predict_generator(testGene, 2, verbose=1)

saveResult("data/membrane/newtest", results)
