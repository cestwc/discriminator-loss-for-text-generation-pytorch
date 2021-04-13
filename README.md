# discriminator-loss-for-text-generation-pytorch
A FastText model discriminator

It could be added as a loss function, side by side with ```torch.nn. NLLLoss```. When a pretrained discriminator model is use, it is supposed to account for a smaller portion of loss, otherwise there may be vanishing gradients.

## Usage
First, you need to have a pretrained model. We understand that a optimized model might not be a good idea for loss function, but we are just providing an option here. To pretrain your model, you need a few more scripts here, to help you align your vocabulary in all models.
```
git clone https://github.com/cestwc/unofficial-torchtext-oov-extension.git
```

Then you can try this. A large portion of codes are from this [tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb), but there are still necessary changes.

```python
from torchtext.legacy import data, datasets

from customized import ENGLISHTEXT
from dLoss import FastText

TEXT = ENGLISHTEXT(include_lengths = True, build_vocab = True) # its 'field name' is 'text'
LABEL = data.LabelField(dtype = torch.float)
```

```python
import random

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split()

LABEL.build_vocab(train_data)
```
Always remember to build your lable 'vocab' according to how many classes you have.

```python
BATCH_SIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    sort_key = lambda x : len(x.text),
    device = device)
	
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

import torch.optim as optim
optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
	"""
	Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
	"""

	#round predictions to the closest integer
	rounded_preds = torch.round(torch.sigmoid(preds))
	correct = (rounded_preds == y).float() #convert into float for division 
	acc = correct.sum() / len(correct)
	return acc

def train(model, iterator, optimizer, criterion):
    
	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for batch in iterator:

		optimizer.zero_grad()

		if isinstance(batch.text, tuple):
			text, text_len = batch.text
		else:
			text = batch.text

		predictions = model(text).squeeze(1)

		loss = criterion(predictions, batch.label)

		acc = binary_accuracy(predictions, batch.label)

		loss.backward()

		optimizer.step()

		epoch_loss += loss.item()
		epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
	epoch_loss = 0
	epoch_acc = 0

	model.eval()

	with torch.no_grad():

		for batch in iterator:
		
			if isinstance(batch.text, tuple):
				text, text_len = batch.text
			else:
				text = batch.text

			if text.shape[0] == 0:
				continue

			predictions = model(text).squeeze(1)

			loss = criterion(predictions, batch.label)

			acc = binary_accuracy(predictions, batch.label)

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```
Then, you may consider use the pretrained model as a discriminator, and freeze it while generating some texts.
```python
from customized import ENGLISHTEXT
from dLoss import DLoss

SRC = ENGLISHTEXT(include_lengths = True, build_vocab = True)
TRG = ENGLISHTEXT(include_lengths = True, build_vocab = True)
```
This customized field helps you maintain a consistency in vocabulary, and thus you may encounter fewer problems when applying the loss model.

```python
INPUT_DIM = len(TRG.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

dloss = DLoss(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX, 'tut3-model.pt').to(device)
```

A better practice would be to freeze the loss model like this
```python
for name, param in dloss.named_parameters():                
	param.requires_grad = False
```

Finally, you can call your loss model like this
```python
loss = dloss(your_output_text)
loss.backward()
```

Again, try to use a proper ratio to balance this loss and the rest, to avoid vanishing gradients.
```
