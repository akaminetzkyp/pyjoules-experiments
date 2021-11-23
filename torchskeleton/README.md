In order to execute [torchskeleton](https://github.com/wbaek/torchskeleton) training, create an environment with the packages in `requirements.txt` and then run the following command:

    python bin/dawnbench/cifar10.py --seed 0xC0FFEE --download > log_dawnbench_cifar10.tsv

Choose the number of epochs and runs with `--epochs` `--runs`.

The results will be in the `torchskeleton/csv` directory.
