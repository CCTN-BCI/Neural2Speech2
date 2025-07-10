


def train_epoch(model, dataloader, optimizer, criterion, device,epoch):
    model.train()
    total_loss = 0
    for batch in dataloader:
        src, tgt, _, _, _, tgt_lens = batch
        src, tgt = src.to(device), tgt.to(device)
        #print('tgt,tgt.shape',tgt,tgt.shape)
        
        optimizer.zero_grad()
        outputs = model(src, tgt[:, :-1]) #outputs = model(src, tgt[:, :-1]) 
        #print(outputs["tokens"],outputs["tokens"].shape)
        targets = {
            'tokens': tgt[:, 1:],  # Shift for teacher forcing
            'length': torch.tensor(tgt_lens, device=device).float() - 1  # Length offset
        }
        
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def predict(model, src, max_len=50, device='cpu'):
    model.eval()
    src = src.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 手动创建SOS（零向量）
        sos_token = torch.zeros(1, 1, model.output_dim).to(device)
        
        # 首先生成预测长度
        memory = model.transformer.encoder(
            model.pos_encoder(model.encoder_embed(src) * math.sqrt(model.d_model))
        )
        pred_length = int(F.softplus(model.length_head(memory.mean(dim=1))).round().item())
        pred_length = min(max(1, pred_length), max_len)
        
        # 从SOS开始生成
        current_token = sos_token
        output_tokens = []
        
        for _ in range(pred_length):
            output = model.transformer.decoder(
                model.pos_decoder(model.decoder_embed(current_token) * math.sqrt(model.d_model)),
                memory,
                tgt_mask=model.generate_square_subsequent_mask(current_token.size(1)).to(device)
            )
            next_token = model.fc_out(output[:, -1:, :])
            output_tokens.append(next_token)
            current_token = next_token
        
        return torch.cat(output_tokens, dim=1), pred_length