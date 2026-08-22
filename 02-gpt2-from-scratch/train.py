import os
import torch
from peft import get_peft_model, PeftModel
from model import MyAdam
from data import get_dataloader, infinite_batches, sft_collate, enc


#checkpoint load for lora
def load_checkpoint_lora(model, chkp_path, lora_config, loading = True):
    if loading:
        # без этого промах по пути молча давал свежий адаптер вместо обученного
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f'No LoRA checkpoint at {chkp_path}. Pass loading=False to start from scratch.')
        model = PeftModel.from_pretrained(model, chkp_path, is_trainable = True)
        print(f'Checkpoint loaded from {chkp_path}.')
    else:
        model = get_peft_model(model, lora_config)
        print('Starting training from scratch.')
    return model


#resuming from checkpoint if it exists
def load_checkpoint(model, optimizer, chkp_path, loading = True):
    if loading and os.path.exists(chkp_path):
        checkpoint = torch.load(chkp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f'Checkpoint loaded. Resuming from step {start_step}.')
        return start_step
    else:
        print('No checkpoint found or loading disabled. Starting training from scratch.')
        return 0


#------------------------------------------------- TRAINING CYCLE

# full training loop (pretrained, sft, lora) with checkpointing
def train_model(num_steps, mode, model, train_data, val_data, device, batch_size = 16, ckpt_path = None, patience = None, min_delta = 1e-3, loading = True):
    train_losses, val_losses = [], []
    best_loss = float('inf')
    patience_counter = 0
    if mode == 'pretrained':
        learning_rate = 1e-3
    elif mode == 'sft':
        learning_rate = 3e-5
    else:
        learning_rate = 1e-4
    # optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = learning_rate)
    optimizer = MyAdam([p for p in model.parameters() if p.requires_grad], lr = learning_rate)
    start_step = load_checkpoint(model, optimizer, ckpt_path, loading = loading)
    total_train_loss = 0.0
    collate_fn = None if mode == 'pretrained' else sft_collate
    train_loader = infinite_batches(get_dataloader(train_data, batch_size, shuffle = True, collate_fn = collate_fn))
    for epoch in range(start_step, start_step + num_steps):
        x, y = next(train_loader)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        total_train_loss += loss.item()
        if epoch % 10 == 9:
            train_losses.append(total_train_loss/10)
            val_loss = evaluate(model, val_data, mode, device, num_iters = 40)
            # early stopping logic
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
                if ckpt_path is not None:
                    torch.save({
                        'step': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f'Checkpoint saved at step {epoch}.')
            else:
                patience_counter += 1
            
            val_losses.append(val_loss)
            
            print(f'"Epoch: {epoch + 1} | Train Loss: {total_train_loss/10:.4f} | Val Loss: {val_loss:.4f}')
            total_train_loss = 0.0

        loss.backward()
        optimizer.step()

        # saving the state of the model, строго после шага, иначе на диск уходит состояние на шаг назад
        if patience is not None and patience_counter >= patience:
            print(f'Early stopping at step {epoch}. Best validation loss: {best_loss:.4f}')
            break
        

    return train_losses, val_losses, best_loss


@torch.no_grad()
def evaluate(model, data, mode, device, batch_size = 16, num_iters = 20):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    collate_fn = None if mode == 'pretrained' else sft_collate
    loader = infinite_batches(get_dataloader(data, batch_size, shuffle = True, collate_fn = collate_fn))
    for _ in range(num_iters):
        x, y = next(loader)
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    avg_loss = total_loss / num_iters
    model.train(was_training)
    return avg_loss


@torch.no_grad()
def complete(model, prompt, device, max_new_tokens = 250, temperature = 0.9, top_k = 50):
    model.eval()
    print(f'Prompt: {prompt}')
    idx = enc.encode(prompt)
    idx = torch.tensor(idx, dtype = torch.long)[None, :].to(device)
    idx = model.generate(idx, max_new_tokens, enc.eot_token, temperature, top_k,
                         stream_decode = lambda t: enc.decode([t]))
    completion = enc.decode(idx[0].tolist())
    return completion
