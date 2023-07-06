lines_seen = set() # keep track of unique lines
with open('train_resnet.log', 'r') as f:
    for line in f:
        if 'Epoch:' in line and 'Total time:' in line:
            print(line)