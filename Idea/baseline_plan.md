# Baseline 方案与失效记录

## 1. 主 baseline 选择

本项目的主 baseline 选择为：

**冻结 `CLIP ViT-B/32` 图像编码器 + 线性分类头**

选择它的理由：

1. 复现成本低，不需要从头训练大型骨干网络。
2. 预训练视觉特征较强，适合作为后续双流融合的语义分支。
3. 对课程项目来说，结果更容易稳定复现。
4. 配合 `CIFAKE` 这类小图像数据集，能够快速完成轻量实验。

该 baseline 的方法来源是 `Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021` 提出的 `CLIP`。结合当前工程实现，实际采用的是 `transformers` 中的 `openai/clip-vit-base-patch32` 视觉编码器，并在其上接一个线性二分类头。

为了让实验对照更完整，当前工程还额外包含两个 CNN baseline：

1. `ResNet18`
   来源于经典残差网络 `He et al., Deep Residual Learning for Image Recognition, CVPR 2016`，作为传统 CNN baseline，用于对比预训练视觉编码器与常规残差网络在小样本和扰动条件下的表现。
2. `MobileNetV3-Small`
   来源于轻量网络 `Howard et al., Searching for MobileNetV3, ICCV 2019`，作为更轻量、更弱的 CNN baseline，用于和 `ResNet18` 形成容量对照，并帮助说明“轻量化”与“鲁棒性”之间的关系。

## 2. Baseline 输入输出

1. 输入：经过统一预处理的 `224 x 224` RGB 图像
2. 图像编码器：冻结参数，不在第一阶段微调
3. 分类头：一层线性层，输出真假二分类 logits
4. 损失函数：`CrossEntropyLoss`

## 3. 第一版训练配置

建议使用下面这套保守配置先跑通：

1. batch size：`32`
2. epoch：`10`
3. optimizer：`AdamW`
4. learning rate：`1e-4`
5. weight decay：`1e-4`
6. early stopping：验证集 `AUC` 连续 `3` 个 epoch 不提升则停止

重点是先稳定得到一组可信结果，而不是一开始就做大规模调参。

## 4. 第一版必须回答的问题

baseline 跑完后，至少要回答下面三个问题：

1. 在 `CIFAKE` 标准测试集上，模型是否能达到明显高于随机猜测的效果？
2. 训练样本规模减小时，性能下降多少？
3. 经过 JPEG 压缩或缩放后，性能又下降多少？
4. 换到外部 benchmark `HybridForensics` 后，性能会下降多少？

如果这三个问题没有明确答案，后续改进模型就没有“改进目标”。

## 5. 你应该重点记录的失效现象

### 失效现象 A：小样本下掉点

典型表现：

1. 训练和同分布测试效果不错
2. 训练样本减少后，AUC 或 Accuracy 明显下降

解释意义：

这说明模型对数据规模比较敏感，鲁棒特征学习能力有限。

### 失效现象 B：对压缩敏感

典型表现：

1. 原始图像上效果较好
2. JPEG 压缩后结果明显变差

解释意义：

这说明模型过度依赖高频细节或脆弱纹理线索，不够稳健。

### 失效现象 C：上采样与格式驱动

典型表现：

1. 仅在固定预处理口径下表现很好
2. 一旦改变上采样或 JPEG 设置，性能明显下降

解释意义：

这说明 baseline 可能利用了预处理模式，而不是学到了真正稳定的真伪判别线索。

### 失效现象 D：跨数据集掉点

典型表现：

1. 在 `CIFAKE` 标准测试集上表现很好
2. 一旦换到 `HybridForensics` 这种外部 benchmark，AUC 和 Accuracy 明显下降

解释意义：

这说明模型对训练数据分布存在依赖，学到的特征不一定能够稳定迁移到更复杂的真实分布。

## 6. 建议记录表

每次 baseline 实验都填一行：

| exp_id | bias_control | train_domain | test_domain | perturbation | acc | auc | f1 | note |
|---|---|---|---|---|---|---|---|---|
| b1 | yes | full train | test | none |  |  |  |  |
| b2 | yes | small train | test | none |  |  |  |  |
| b3 | yes | full train | test | jpeg85 |  |  |  |  |
| b4 | yes | full train | test | jpeg75 |  |  |  |  |
| b5 | yes | full train | test | resize |  |  |  |  |
| b6 | yes | CIFAKE train | HybridForensics | none |  |  |  |  |

## 7. Baseline 阶段的最低交付物

完成 baseline 阶段后，至少应该拿到：

1. 一张 `full train vs small train` 性能对比表
2. 一张压缩鲁棒性结果表
3. 一张外部 benchmark 结果表
4. 一段对“小样本训练 + 扰动条件 + 跨数据集”下失效原因的文字总结

这三项会直接成为答辩和报告中的“问题动机”部分。

## 8. 当前外部 benchmark 记录

目前已经补充了 `HybridForensics` 作为第二 benchmark。该数据集共 `10000` 张图像，其中 `5000 real + 5000 fake`，假图同时包含 `ProGAN`、`StyleGAN3`、`SDXL` 和 `Midjourney`，因此比 `CIFAKE` 更适合作为外部测试集。

当前首轮外部测试结果为：

1. 模型：`MobileNetV3-Small`
2. checkpoint：`baseline/outputs/mobilenet_v3_small/one_epoch_test/best.pt`
3. `Accuracy = 0.5231`
4. `AUC = 0.5919`
5. `F1 = 0.6672`

这个结果说明：

1. 轻量 CNN baseline 在 `CIFAKE` 上训练后，迁移到外部 benchmark 时性能明显下降
2. 仅在单一数据集上取得高分，并不意味着模型具备良好的跨数据集泛化能力
3. 后续引入 `SRM` 或双流融合是有必要的，因为当前结果已经暴露出明显的数据集依赖
## 9. 本文档对应的执行结论

当前 baseline 口径已经固定为：

1. 主 baseline：`CLIP ViT-B/32 + linear head`
2. 对照 baseline：`ResNet18`、`MobileNetV3-Small`
3. 外部 benchmark：`HybridForensics`
4. 核心观察：小样本性能下降、压缩敏感性、预处理影响、跨数据集掉点
5. 交付要求：至少拿到 4 组能支撑改进动机的结果
