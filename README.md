The model can be trained in distributed mode with multiple GPUs on a single machine. Option --num-gpu represents the number of GPUs you want to use for training. Option --kv-store can be set to 'local' or 'device'. Refer to [official docs](https://mxnet.incubator.apache.org/api/python/kvstore/kvstore.html#mxnet.kvstore.create) for details.