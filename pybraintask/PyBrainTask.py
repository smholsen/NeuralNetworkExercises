from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(1, 1)

# Add samples that matches the description in the task
ds.addSample((1,), (1,))
ds.addSample((2,), (2,))
ds.addSample((3,), (3,))
ds.addSample((4,), (4,))
ds.addSample((5,), (5,))
ds.addSample((6,), (6,))
ds.addSample((7,), (7,))
ds.addSample((8,), (8,))

net = buildNetwork(1, 2, 1, hiddenclass=TanhLayer)

trainer = BackpropTrainer(net, ds)

trainer.trainUntilConvergence(verbose=False, validationProportion=0.15,
                              maxEpochs=1000, continueEpochs=10)

print(net.activate((1, ))[0])
print(net.activate((2, ))[0])
print(net.activate((3, ))[0])
print(net.activate((4, ))[0])
print(net.activate((5, ))[0])
print(net.activate((6, ))[0])
print(net.activate((7, ))[0])
print(net.activate((8, ))[0])

print(net.activate((0.5, ))[0])
print(net.activate((-4.1, ))[0])
print(net.activate((10, ))[0])
print(net.activate((100, ))[0])
print(net.activate((8.0, ))[0])

