import torch


def print_status(current_status, total, more_info="", bar_length = 30):
    print(f"\r {current_status}/{total} [", end="")
    printed_progress = False
    for current in range(bar_length):
        if current / bar_length < current_status / total:
            print("=", end="")
        else:
            if(not printed_progress):
                print(">",end="")
            else:
                print(".", end="")
            printed_progress = True
    print("]"+more_info, end="")
        
        
def train_model(model, kb, epochs=100,lr=0.001):

    criteria = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    patience = 5
    min_delta = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}")
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        training_iterations_count = len(kb.loaders[0])

        # Training loop
        for data, labels in kb.loaders[0]:
            optimizer.zero_grad()
            predictions = model(data)
            loss = criteria(predictions, labels)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += loss.item()
            num_batches += 1


            more_info = f" - loss: {current_loss}"

            print_status(num_batches, training_iterations_count, more_info=more_info)

        print()        

        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        for data, labels in kb.val_loaders[0]:
            with torch.no_grad():
                predictions = model(data)
                val_loss = criteria(predictions, labels)
                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        # Early stopping logic
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

        model.train()

def predict(model, x):

    if hasattr(model, 'predict'):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        preds = model.predict(x)
        return torch.tensor(preds, dtype=torch.float32)
    
    model.eval() 

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    elif x.dtype != torch.float32:
        x = x.float()

    with torch.no_grad():  
        probs = model(x)
        preds = (probs > 0.5).float()

    return preds.cpu().numpy()

def cache_datasets(map):
    for key, value in map.items():
        value.to_csv(f"./training_cache/{key}.csv", index=False)