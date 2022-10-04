import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Hyper Parameters
BATCH_SIZE = 1
LR = 0.01                   # learning rate
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 500

ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
 
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc1 = nn.Linear(in_dim, out_dim, bias=True)
        self.fc2 = nn.Linear(out_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=True)
 
    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        # z为发出结点的特征向量，e为边权，这里就是注意力，整体来看，比如发出节点为i，接收节点为j，这条消息就是i的特征向量和i->j的注意力
        return {'z' : edges.src['z'], 'e' : edges.data['e']}
 
    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        # nodes.mailbox表示结点收到的信息（即特征向量和注意力）'e'为注意力值，'z'为特征向量
        alpha = F.softmax(nodes.mailbox['e'], dim=1) # alpha为注意力矩阵
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}
 
    def forward(self, h):
        # 公式 (1)
        z = self.fc1(h)
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
 
    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs))
    
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。此外输出层只有一个头。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        self.out = nn.Linear(hidden_dim * num_heads, out_dim, bias=False)
        #self.layer2 = nn.Linear(hidden_dim * num_heads, out_dim, bias=True)
 
    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        #h = self.layer2(h)
        h = self.out(h)
        return h

class GADQN(object):
    def __init__(self, g, N_STATES, N_ACTIONS):
        self.eval_net = GAT(g, 
          in_dim=N_STATES, 
          hidden_dim=8, 
          out_dim=N_ACTIONS, 
          num_heads=8)
        
        self.target_net = GAT(g, 
          in_dim=N_STATES, 
          hidden_dim=8, 
          out_dim=N_ACTIONS, 
          num_heads=8)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = []     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
    
    def save_model(self):
        torch.save(self.eval_net.state_dict(),f'model_{time.time()}')
    
    def choose_action(self, x, epsilon):

        x = torch.FloatTensor(x)
        #x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        
        logits = self.eval_net.forward(x)
        logp = F.log_softmax(logits, 1)
        action = torch.max(logp, 1)[1].data.numpy().reshape(-1,1)

        for i in range(len(action)):
            if np.random.uniform() < epsilon:
                action[i][0] = np.random.randint(0, self.N_ACTIONS)

        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % MEMORY_CAPACITY
        if len(self.memory) <= index:
            self.memory.append((s, a, r, s_))
        else:
            self.memory[index] = (s, a, r, s_)
        self.memory_counter += 1

    def get_attention_record(self, shape):
        self.attention_record.forward(shape)

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_memory = random.sample(self.memory, BATCH_SIZE)

        b_s = []
        b_a = []
        b_r = []
        b_s_ = []
        for s,a,r,s_ in b_memory:
            b_s.append(s)
            b_a.append(a)
            b_r.append(r)
            b_s_.append(s_)
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s[0]).gather(1, b_a[0])  # shape (batch, 1) 
        q_next = self.target_net(b_s_[0]).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(-1, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss = loss
        
    def load_model(self):
        self.eval_net.load_state_dict(torch.load('model_1653979481.640722'))