# Flow chart of how this method works

**Steps**
1. Cross Attention across MSA dimension into n_kv_heads 
   1. input: B x M x S x D -> B x S x M - 1 x D, take top seq for B x S x D
   2. Channel projection: B x S x D -> B x S x C x 1 x D
   3. Q projection: B x S x M x D -> BS x M x HD -> BS x H x M x D_attn
   4. Attention: B x S x D
2. Swap: B x S x D
3. Channel Proj: B x C x S x D
4. Conv: B x C x S x D
5. Hydra: B x C x S x D (heads = C)
6. Conv: B x C x S x D
7. Channel Proj: B x S x D
8. Full self-Attention across MSA dimension
   1. Add to old MSA
9. MLP Block