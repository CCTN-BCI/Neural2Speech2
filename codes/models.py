class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Seq2SeqTransformer(nn.Module):
    def __init__(self, input_dim=42, output_dim=1024, d_model=256, nhead=8, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        self.encoder_embed = nn.Linear(input_dim, d_model)
        self.decoder_embed = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        
        # Enhanced length prediction head
        self.length_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src, tgt=None, max_len=50, is_inference=False):
        #print('src.shape',src.shape)
        src = self.encoder_embed(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #print('src.shape',src.shape)
        if is_inference:
            return self._inference_forward(src, max_len)
        
        tgt = self.decoder_embed(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        sz = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(sz).to(tgt.device)
        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        
        output_tokens = self.fc_out(output)
        # Predict length from mean of encoder output
        output_length = F.softplus(self.length_head(src.mean(dim=1)))
            
        return {
            'tokens': output_tokens,
            'length': output_length.squeeze(-1)
        }
    
    def _inference_forward(self, src, max_len):
        memory = self.transformer.encoder(src)
        batch_size = src.size(0)
        
        # Predict sequence length first
        #length_embed = self.length_head(memory.mean(dim=1))
        length_embed = self.length_head(src.mean(dim=1))
        pred_length = int(F.softplus(length_embed).round().item())
        pred_length = min(max(1, pred_length), max_len)  # Clamp to valid range
        
        # Generate sequence based on predicted length
        tgt = torch.zeros(batch_size, 1, self.output_dim).to(src.device)
        output_tokens = []
        
        for _ in range(pred_length):
            tgt_embed = self.decoder_embed(tgt) * math.sqrt(self.d_model)
            tgt_embed = self.pos_decoder(tgt_embed)
            
            output = self.transformer.decoder(
                tgt_embed, 
                memory,
                tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
            )
            
            next_token = self.fc_out(output[:, -1:, :])
            output_tokens.append(next_token)
            tgt = torch.cat([tgt, next_token], dim=1)
        
        output_tokens = torch.cat(output_tokens, dim=1)
        return output_tokens, pred_length
class DynamicSequenceLoss(nn.Module):
    def __init__(self, token_weight=1, length_weight=1):
        super().__init__()
        self.token_weight = token_weight
        self.length_weight = length_weight
        self.token_loss = nn.KLDivLoss(reduction='batchmean')  
        self.length_loss = nn.HuberLoss()

    def forward(self, preds, targets):
        # 对预测值取log_softmax（KL散度要求）
        pred_log_probs = F.log_softmax(preds['tokens'], dim=-1)
        # 确保目标是有效的概率分布
        target_probs = F.softmax(targets['tokens'], dim=-1)
        # Token-level KL散度损失
        token_loss = self.token_loss(
            pred_log_probs,  # 输入需要是log probabilities
            target_probs     # 目标需要是probabilities
        )
        
        # Length prediction loss (保持不变)
        length_loss = self.length_loss(
            preds['length'], 
            targets['length']
        )
        
        total_loss = (self.token_weight * token_loss +
                     self.length_weight * length_loss)
        
        return {
            'total': total_loss,
            'token': token_loss,
            'length': length_loss
        }