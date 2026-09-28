def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for batch in dataloader:
        question_ids = batch["question_ids"].to(device)
        context_ids = batch["context_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(
            question_ids,
            context_ids,
        )

        loss = criterion(
            logits.reshape(-1, 3),
            labels.reshape(-1),
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
