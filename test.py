import lmdb
path = "/linkhome/rech/gennip01/ura93tx/storage/datasets/Adobe240/train_lr.lmdb"

env = lmdb.open(path, readonly=True, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    for i, (k, v) in enumerate(cursor):
        print(k.decode())
        if i >= 5:
            break
env.close()
