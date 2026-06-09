import torch
import torch.nn as nn
import torch.nn.functional as F
from _3_predictmodel.utils.utils import initialize_weights


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=512, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Mish()
        ]

        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]

        if dropout:
            self.attention_a.append(nn.Dropout(0.3))
            self.attention_b.append(nn.Dropout(0.3))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class HamiltonianNetwork(nn.Module):
    def __init__(self, feature_dim, time_steps=3, delta_t=0.01, evolution_type='simple'):
        super(HamiltonianNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.delta_t = delta_t
        self.evolution_type = evolution_type

        # 处理奇数维度
        if feature_dim % 2 != 0:
            self.odd_dim = True
            self.main_dim = feature_dim - 1
            self.half_dim = self.main_dim // 2
        else:
            self.odd_dim = False
            self.main_dim = feature_dim
            self.half_dim = feature_dim // 2

        # 创建固定的辛矩阵 - 不需要训练
        if self.evolution_type == 'symplectic':
            self.register_buffer('symplectic_matrix', self._create_symplectic_matrix())

        # 可学习的演化强度参数（标量，影响整体演化程度）
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _create_symplectic_matrix(self):
        """创建标准辛矩阵 J = [[0, I], [-I, 0]]"""
        I = torch.eye(self.half_dim)
        zero = torch.zeros(self.half_dim, self.half_dim)
        upper = torch.cat([zero, I], dim=1)
        lower = torch.cat([-I, zero], dim=1)
        J = torch.cat([upper, lower], dim=0)
        return J

    def _symplectic_evolution(self, x):
        """基于辛矩阵的哈密顿演化"""
        # x: [batch_size, main_dim]
        for _ in range(self.time_steps):
            # 辛几何演化: dx/dt = J * grad_H(x)
            # 这里使用简单的二次哈密顿函数 H = 0.5 * x^T * x
            grad_H = x  # 对于二次哈密顿函数，梯度就是x本身

            # 辛变换
            dx = torch.matmul(grad_H, self.symplectic_matrix.T) * self.delta_t
            x = x + dx

        return x

    def forward(self, x):
        # 分离主要特征和奇数维度
        if self.odd_dim:
            main_features = x[:, :-1]
            odd_feature = x[:, -1:]
        else:
            main_features = x
            odd_feature = None

        evolved_main = self._symplectic_evolution(main_features)
        # 重新组合奇数维度
        if self.odd_dim:
            # 奇数维度保持不变
            evolved_features = torch.cat([evolved_main, odd_feature], dim=1)
        else:
            evolved_features = evolved_main

        # 使用可学习参数控制演化强度
        # α=0时输出等于输入，α=1时完全演化
        output = x + self.alpha * (evolved_features - x)

        return output


class NewModel_3(nn.Module):
    def __init__(self, conf, instance_loss_fn=nn.CrossEntropyLoss()):
        super(NewModel_3, self).__init__()
        n_classes = conf.n_class

        self.use_hamiltonian = getattr(conf, 'use_hamiltonian')
        hamiltonian_time_steps = getattr(conf, 'hamiltonian_time_steps', 3)
        hamiltonian_delta_t = getattr(conf, 'hamiltonian_delta_t', 0.01)
        hamiltonian_evolution_type = getattr(conf, 'hamiltonian_evolution_type', 'symplectic')

        self.dropout = getattr(conf, 'dropout')
        self.size_arg = getattr(conf, 'size_arg')
        self.gate = getattr(conf, 'gate')
        self.size_dict = {"small": [conf.D_feat, conf.D_inner, 128], "big": [conf.D_feat, 512, 384]}

        size = self.size_dict[self.size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if self.dropout:
            fc.append(nn.Dropout(0.1))
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=self.dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # 改进的哈密顿网络
        if self.use_hamiltonian:
            self.hamiltonian_network = HamiltonianNetwork(
                feature_dim=size[1],
                time_steps=hamiltonian_time_steps,
                delta_t=hamiltonian_delta_t,
                evolution_type=hamiltonian_evolution_type
            )

        self.classifiers = nn.Linear(size[1], n_classes)

        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = getattr(conf, 'k_sample')
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = False
        if conf.n_class > 2:
            self.subtyping = True

        initialize_weights(self)


    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False,
                return_attention=False):
        """
        前向传播函数，增加了返回attention score的功能

        Args:
            h: 输入特征
            label: 标签（用于实例评估）
            instance_eval: 是否进行实例评估
            return_features: 是否返回特征
            attention_only: 是否只返回attention
            return_attention: 是否返回attention score（新增参数）
        """
        A, h = self.attention_net(h[0])  # NxK
        A = torch.transpose(A, -1, -2)  # KxN

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A_raw, dim=-1)  # softmax over N

        # 在特征聚合前应用哈密顿网络进行特征演化
        if self.use_hamiltonian:
            with torch.set_grad_enabled(self.training):  # 训练时开启梯度，评估时关闭
                h = self.hamiltonian_network(h)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)  # 注意力加权聚合特征
        logits = self.classifiers(M)

        # 根据不同的返回要求构建返回值
        if instance_eval:
            if return_attention:
                return logits, total_inst_loss, A  # 返回attention score
            else:
                return logits, total_inst_loss
        else:
            if return_attention:
                return logits, A  # 返回attention score
            else:
                return logits