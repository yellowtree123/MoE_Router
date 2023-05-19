# MoE_Router

基于多种转发方式的MoE模型。

目前支持Expert模型结构为MLP以及CNN的模型，支持数据集为MNIST以及CiFar10。使用"--use_moe"决定是否使用混合专家网络。

实现的转发方法有：

1. Noisy_Topk&Topk：支持$1 \leq   k \leq E$，其中$E$为Expert数量。
2. Anneal：退火使用一个线性scheduler根据输入的最高温度以及最低温度线性变化，对于softmax函数进行逐退火。
3. Hash：对于所有的输入训练样本赋予随机的固定性转发专家
4. BASE：基于线性规划的拍卖算法，在训练对于logits给出均衡分布的转发信号。
5. TO DO:REINFORCE：强化学习的损失函数

