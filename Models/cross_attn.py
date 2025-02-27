import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """
    A cross-attention block that injects text features into image features.
    
    This module applies a multi-head attention mechanism where image features
    (queries) attend to text embeddings (keys and values). The text features
    are projected to match the image feature dimension before attention.
    
    Args:
        feat_dim (int): Dimension of image features.
        text_dim (int, optional): Dimension of text embeddings. Defaults to 256.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
    """

    def __init__(self, feat_dim, text_dim=256, num_heads=8):
        """
        Initializes the cross-attention block with a layer normalization,
        a linear projection for text embeddings, and a multi-head attention module.
        This is exactly as attention is all you need implements it.

        Args:
            feat_dim (int): Dimension of image features.
            text_dim (int, optional): Dimension of text embeddings. Defaults to 256.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)  # for normalising image features
        self.text_proj = nn.Linear(text_dim, feat_dim)  # projection of text embeddings to image feature dimension
        self.attention = nn.MultiheadAttention(feat_dim, num_heads)  # multi-head attention mechanism

    def forward(self, img_feat, text_emb):
        """
        Applies cross-attention between image features and text embeddings.
        
        The image features act as the queries, while the text embeddings 
        (after projection) act as both keys and values in the attention mechanism.
        A skip connection is added after attention computation.

        Args:
            img_feat (torch.Tensor): Image features of shape (B, H*W, C), where
                                     B is the batch size, H*W is the number of patches/spatial tokens, 
                                     and C is the feature dimension.
            text_emb (torch.Tensor): Text embeddings of shape (B, L, D), where
                                      L is the sequence length (number of text tokens),
                                      and D is the embedding dimension.

        Returns:
            torch.Tensor: Updated image features with injected text guidance,
                          maintaining the shape (B, H*W, C).
        """
        # projection
        text_feat = self.text_proj(text_emb)  # (B, L, feat_dim)

        # normalisation
        img_feat = self.norm(img_feat)  # (B, H*W, C)

        # transpose the inputs for MultiheadAttention (seq_len, batch, feature_dim)
        img_feat = img_feat.transpose(0, 1)  # (H*W, B, C)
        text_feat = text_feat.transpose(0, 1)  # (L, B, C)
        
        # attention (this step helps in injection)
        attn_out, attn_weights = self.attention(
            query=img_feat,  # (B, H*W, C)
            key=text_feat,   # (B, L, C)
            value=text_feat,  # (B, L, C)
            need_weights=True,
            average_attn_weights=False
        )
        
        # attn_weights shape: (B, num_heads, H*W, L)
        attn_weights = attn_weights.mean(dim=1)  # average across heads
        attn_weights = attn_weights.squeeze(1)  # remove the heads dimension
        img_feat = img_feat.transpose(0, 1)  # (B, H*W, C)
        attn_out = attn_out.transpose(0, 1)  # (B, H*W, C)
        
        # adding skip connection
        return img_feat + attn_out, attn_weights  # both (B, H*W, C)