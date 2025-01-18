# Description
The code provides a simple implementation of behaviour cloning for the various environments in MineRL (in my case, MineRLTreechop-v0).
I have created a custom dataloader (`data_pipeline.py` and `buffered_batch_iter.py`) to allow support with MineRL v1.0 as well. However, due to issues with `env.render()` This code currently does not support MineRL v1.0 yet.
# Installation
To be able to run this code, follow the [installation steps](https://minerl.readthedocs.io/en/v0.4.4/tutorials/index.html) for MineRL v0.4.4. Also install the necessary dataset from MineRL from [here](https://zenodo.org/records/12659939).
# Run
Uncomment train() in main if you wish to train the model yourself.
Run `behavioural_cloning.py` for training your model and testing it afterwards.
# Credits
I was able to write this code with the help of a useful [template](https://github.com/minerllabs/getting-started-tasks/blob/main/behavioural_cloning.py) provided by minerllabs.
