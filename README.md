# 3D Object Classification with PointNet


## Hyper Parameters tuning with Ray

The `tune.py` script contains all the required code to run K samples from a space of hyperparameters.
These are defined by a dictionary called config in the __main__ function at the end.
If more hyperparameters are needed, one must also modify the train_loop_per_worker function to
use them in the training loop. If one needs to customize the search look in the __main__ function where ray is used.

The model is imported as `PoinNet` in `src.models.pointnet` at the beginning of the script. If you wish to use 
a different model, either modify the class in `src.models.pointnet`, or import a different one, and make sure
that the train loop uses the intended model.

Before running the `tune.py` script, make sure that you run this in the CLI:
```
ray start --head
```
to start the head node of the ray cluster on the current machine. Then, for each of the VMs
that you want to use as workers, run this in their CLI: (ray start will usually write you this line when you launch it)

```
ray start --address='HEAD_NODE_VM_IP:6379'
```

This way, they will connect to the head node and act as workers.  
Finally, you can go back to the head node and run the script `tune.py`; 
you can use nohup and & to have it run in the background so you can close SSH
and have it print the output in a file called nohup.out like so:

```
nohup python tune.py &
```

This is because the Python script looks for the head node on the current machine
in order to launch the loop in a distributed manner.
When you are done with everything, you can shut down the ray nodes with 
```
ray stop
```

At any time after you started the head node, you can control the status of the cluster by checking the dashboard (by default found at port 8265) on the head node VM.

### Shared FS

Right now the way the various train loops access the files is via a shared file system on the head node. Read more [here](https://wiki.ubuntu-it.org/Server/Nfs). The dataset object uses a metadata file that points to this location. 
