import torch
import torch.nn.functional as F


class DropoutNet_model(torch.nn.Module):
    def __init__(self, env, dataset):
        super(DropoutNet_model, self).__init__()
        self.env = env
        self.n_user = dataset.n_user
        self.m_item = dataset.m_item
        self.free_emb_dimension = self.env.args.free_emb_dimension

        self.image_feat = torch.tensor(dataset.image_feat, dtype=torch.float32).to(self.env.device)
        self.image_feat = F.normalize(self.image_feat)
        self.text_feat = torch.tensor(dataset.text_feat, dtype=torch.float32).to(self.env.device)
        self.text_feat = F.normalize(self.text_feat)

        self.user_emb = torch.nn.Embedding(self.n_user, self.free_emb_dimension)
        self.image_linear = torch.nn.Linear(self.image_feat.shape[1], self.free_emb_dimension)
        self.text_linear = torch.nn.Linear(self.text_feat.shape[1], self.free_emb_dimension)

        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        self.to(self.env.device)

    def forward(self, drop_prob=0.0):
        user_emb = self.user_emb.weight

        image_feat = self.image_feat
        text_feat = self.text_feat

        if self.training and drop_prob > 0:
            # 以概率 drop_prob 独立丢弃每个模态
            img_mask = (torch.rand(image_feat.shape[0], 1, device=self.env.device) > drop_prob).float()
            txt_mask = (torch.rand(text_feat.shape[0], 1, device=self.env.device) > drop_prob).float()
            image_feat = image_feat * img_mask
            text_feat = text_feat * txt_mask

        image_emb = self.image_linear(image_feat)
        text_emb = self.text_linear(text_feat)
        item_emb = (image_emb + text_emb) / 2

        return user_emb, item_emb
