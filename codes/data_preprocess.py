def select_electrodes_for_hg(all_hgs, all_electrodes):
    res = np.zeros((all_hgs.shape[0], len(all_electrodes)))
    for i in range(len(all_electrodes)):
        res[:,i] = all_hgs[:,all_electrodes[i]]
    return res

class Dataset_LSM_resynthesis(Dataset):
    def __init__(self, fname_hg_lat, fname_timepoint, subject_name, fname_electrodes,
                 train=True, fname_save_text_features='./cache/LLM/pre-trained_tts_features.pkl', N=50):
        self.all_electrodes = sio.loadmat(fname_electrodes)
        self.train = train
        self.N = N
        with open(fname_hg_lat, 'rb') as f:
            feat_mat = pickle.load(f)
        self.all_timepoints = sio.loadmat(fname_timepoint)
        self.all_hgs = torch.FloatTensor(select_electrodes_for_hg(feat_mat['hg'], self.all_electrodes[subject_name][0]))
        with open(fname_save_text_features, 'rb') as f:
            self.all_text_features = pickle.load(f)
            self.sos_token = (torch.zeros_like(self.all_text_features[0][0][0:1])).unsqueeze(dim = 0)
            print(self.sos_token.shape)
        if self.train:
            self.length = len(self.all_text_features) - self.N
        else:
            self.length = self.N
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.train:
            idx = idx
        else:
            idx = 0 - idx
        idx_begin = self.all_timepoints['times_begin'][0][idx]
        idx_end = self.all_timepoints['times_end'][0][idx]
        data = torch.Tensor(self.all_hgs[idx_begin:idx_end])
        
        text_label = self.all_text_features[idx] 
        
        # 添加SOS到开头
        text_label_with_sos = torch.cat([
            self.sos_token.to(text_label.device),  # 添加SOS
            text_label                             # 原始序列
        ], dim=1)
        
        return data, text_label#text_label_with_sos

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.squeeze(0), y.squeeze(0)


def collate_fn(batch):
    """
    处理批次数据的函数，特点：
    1. 只在target序列开头添加**一个**SOS（零向量）
    2. 其余填充部分用EOS（最后一个有效token）填充
    3. 生成对应的padding mask
    """
    src_batch, tgt_batch = zip(*batch)
    
    # ======================
    # 1. 处理source序列（不变）
    # ======================
    src_lens = [len(x) for x in src_batch]
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True)
    src_padding_mask = torch.zeros_like(src_padded[:, :, 0], dtype=torch.bool)
    for i, l in enumerate(src_lens):
        src_padding_mask[i, l:] = True  # 填充位置为True

    # ======================
    # 2. 处理target序列
    # ======================
    # 2.1 为每个target添加**一个**SOS（零向量）
    tgt_with_sos = []
    for seq in tgt_batch:
        sos_token = torch.zeros_like(seq[0:1])  # 创建SOS（形状[1, feature_dim]）
        tgt_with_sos.append(torch.cat([sos_token, seq], dim=0))  # [SOS, T1, T2..., Tn]
    
    # 2.2 计算添加SOS后的长度
    tgt_lens = [len(x) for x in tgt_with_sos]
    max_tgt_len = max(tgt_lens)
    feature_dim = tgt_with_sos[0].shape[-1]
    
    # 2.3 初始化填充后的tensor
    tgt_padded = torch.zeros(len(tgt_with_sos), max_tgt_len, feature_dim)
    tgt_padding_mask = torch.zeros(len(tgt_with_sos), max_tgt_len, dtype=torch.bool)
    
    # 2.4 填充数据（用EOS填充）
    for i, (seq, seq_len) in enumerate(zip(tgt_with_sos, tgt_lens)):
        tgt_padded[i, :seq_len] = seq
        if seq_len < max_tgt_len:
            # 获取EOS（最后一个有效token，即原序列的最后一个token）
            eos_token = seq[-1]  # 形状[feature_dim]
            # 扩展为[max_tgt_len-seq_len, feature_dim]
            padding_tokens = eos_token.unsqueeze(0).expand(max_tgt_len - seq_len, -1)
            tgt_padded[i, seq_len:] = padding_tokens
        tgt_padding_mask[i, seq_len:] = True  # 填充位置为True

    return src_padded, tgt_padded, src_padding_mask, tgt_padding_mask, src_lens, tgt_lens
