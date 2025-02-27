import torch
import torch.nn as nn

def contrastive_loss(img_emb, word_emb, sent_emb, temperature = 0.07):
    """
    Compute contrastive loss between the three embeddings.
    
    Args:
        img_emb: Image embeddings of shape (batch_size, num_patches, dim) or (batch_size, dim)
        word_emb: Word embeddings of shape (batch_size, seq_len, dim) or (batch_size, dim)
        sent_emb: Sentence embeddings of shape (batch_size, 1, dim) or (batch_size, dim)
        
    Returns:
        Total contrastive loss
    """
    # if inputs are 3D, take mean along sequence dimension
    if len(img_emb.shape) == 3:
        img_emb = torch.mean(img_emb, dim=1)  # (batch_size, dim)
    if len(word_emb.shape) == 3:
        word_emb = torch.mean(word_emb, dim=1)  # (batch_size, dim)
    if len(sent_emb.shape) == 3:
        sent_emb = sent_emb.squeeze(1)  # (batch_size, dim)
        
    img_emb = nn.functional.normalize(img_emb, dim=-1)
    word_emb = nn.functional.normalize(word_emb, dim=-1)
    sent_emb = nn.functional.normalize(sent_emb, dim=-1)
    
    sim_img_word = torch.matmul(img_emb, word_emb.t()) / temperature
    sim_img_sent = torch.matmul(img_emb, sent_emb.t()) / temperature
    
    labels = torch.arange(img_emb.size(0)).to(img_emb.device)
    
    loss_img_word = nn.CrossEntropyLoss()(sim_img_word, labels)
    loss_img_sent = nn.CrossEntropyLoss()(sim_img_sent, labels)
    
    total_loss = loss_img_word + loss_img_sent
    return total_loss