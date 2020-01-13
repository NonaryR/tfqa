import ujson as json
import gc
from tqdm import tqdm

import torch
from apex import amp
from torch.utils.data import DataLoader
import numpy as np

from metrics import eval_model
from data import TextDataset, JsonChunkReader
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

def loaders(DATA, convert_fn, size, chunksize, batch_size, eval_collate_fn):
    reader = JsonChunkReader(DATA, convert_fn, chunksize=chunksize)
    loaders = []
    
    for examples in tqdm(reader, total=int(np.ceil(size / chunksize)), desc="getting validation data"):
        
        dataset = TextDataset(examples)
        loader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=eval_collate_fn,
                            pin_memory=True)
        loaders.append(loader)
        # break
    
    return loaders


def run(model, TRAIN_DATA, DEV_DATA, logs, optimizer, scheduler, loss_fn, collate_fn, eval_collate_fn, convert_fn, \
          n_epochs, batch_size, accumulation_steps, chunksize, train_size, val_size, eval_="do"):
    
    eval_ = True if eval_ == "do" else False

    if eval_:
        val_loaders = loaders(DEV_DATA, convert_fn, val_size, chunksize, batch_size, eval_collate_fn)

    for epoch in range(n_epochs):

        global_step = 0
        model.train(True)
        train_reader = JsonChunkReader(TRAIN_DATA, convert_fn, chunksize=chunksize)
        
        for index, examples in enumerate(tqdm(train_reader, total=int(np.ceil(train_size / chunksize)), desc=f"training-epoch-{epoch+1}")):
            
            train_dataset = TextDataset(examples)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                      shuffle=True, collate_fn=collate_fn)
            
            for x_batch, y_batch in train_loader:
                x_batch, attention_mask, token_type_ids = x_batch
                y_true = (y.cuda(non_blocking=True) for y in y_batch)

                y_pred = model(x_batch.cuda(non_blocking=True),
                               attention_mask=attention_mask.cuda(non_blocking=True),
                               token_type_ids=token_type_ids.cuda(non_blocking=True))
                
                loss = loss_fn(y_pred, y_true)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                if (global_step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                global_step += 1

            if (index+1) % 60 == 0 and eval_:
                result, model = eval_model(model, val_loaders, f"index {index+1} from {epoch+1} epoch")
                result["epoch"] = epoch
                result["index"] = index+1
                with open(f'{logs}/result.txt', 'a') as outfile:
                    json.dump(result, outfile, indent=4)

                # break

            del train_dataset
            # break

        if eval_:
            result, model = eval_model(model, val_loaders, f"epoch {epoch+1}")
            result["epoch"] = epoch
            result["index"] = "end of epoch"
            with open(f'{logs}/result.txt', 'a') as outfile:
                json.dump(result, outfile, indent=4)

        torch.save(model.module.state_dict(), f"{logs}/model-{epoch+1}-epoch.bin")
        
        del train_reader
        gc.collect()
        
