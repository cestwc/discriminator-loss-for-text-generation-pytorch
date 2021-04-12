import torch.nn as nn
import torch.nn.functional as F

class DLoss(nn.Module):
	
	def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx, directory):

		super().__init__()
		
		self.model = FastText(vocab_size, embedding_dim, output_dim, pad_idx)
		
		self.model.load_state_dict(torch.load(directory))
		
	def forward(self, text):
		
		with torch.no_grad():
		
			predictions = self.model(text).squeeze(1)
		
		return torch.mean(torch.sigmoid(predictions))
	

class FastText(nn.Module):
	def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):

		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

		self.fc = nn.Linear(embedding_dim, output_dim)

	def forward(self, text):

		#text = [sent len, batch size]

		text = text.masked_fill(text >= self.embedding.num_embeddings, 0)  # replace OOV words with <UNK> before embedding

		embedded = self.embedding(text)

		#embedded = [sent len, batch size, emb dim]

		embedded = embedded.permute(1, 0, 2)

		#embedded = [batch size, sent len, emb dim]

		pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 

		#pooled = [batch size, embedding_dim]
		
		return self.fc(pooled)
