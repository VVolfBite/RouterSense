# Future EP Weight and Communication Contract

Deployment preparation requires every node to keep a local copy of the model checkpoint, tokenizer, and config. Nodes do not fetch checkpoints from each other during inference.

Future multi-GPU execution will load local weights on each rank, then apply expert placement and communication to route token hidden states and expert outputs. The runtime may replicate or partition non-expert parameters later, but this phase does not implement that design.

During future inference, the only cross-node transfers are token hidden states, routing/token metadata, and expert outputs. Complete checkpoints, model parameter files, and expert weight files are not exchanged between nodes at inference time.
