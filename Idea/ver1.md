# 项目题目

**面向资源受限场景的 AI 生成图像检测：基于 CIFAKE 与双流特征融合的轻量化方法**

**一句话创新点：** 在小规模可训练数据集上，融合预训练视觉语义特征与底层取证特征，并系统评估模型在压缩与缩放扰动下的鲁棒性。

## 1. 项目背景与意义

随着 GAN 和扩散模型的发展，AI 生成图像已经越来越逼真，传统依赖局部纹理伪影的检测器在新型生成模型上容易失效。图像真伪检测不仅是一个有现实意义的问题，也与课程中的深度学习、表示学习、泛化能力和模型鲁棒性等主题直接相关，因此具有较强的研究价值。

本项目关注一个更具体、也更符合现实约束的问题：**在有限显存和存储条件下，如何构建一个既能有效区分真实图像与 AI 图像、又能在压缩和缩放扰动下保持稳定性能的轻量检测器。** 这比单纯做一个“跑大数据集的真假二分类器”更符合课程项目对技术合理性和可完成度的要求。

## 2. 核心问题

我们希望回答以下问题：

1. 为什么很多 AI 图像检测模型在资源受限或输入受扰动时性能会快速下降？
2. 在统一图像大小和压缩格式后，哪些特征仍然能够稳定地区分真实图像与 AI 图像？
3. 能否构建一个**轻量但有效**的融合模型，同时利用高层语义信息和底层取证特征，提高模型在后处理扰动下的稳健性？

## 3. 项目创新点

为了满足课程要求中对 `Novelty` 的强调，本项目不打算只复现已有方法，而是在以下几个方面做改进：

1. **偏差控制的数据设置**  
   不直接使用原始数据分布，而是统一图像分辨率和压缩质量，尽量避免模型仅学习到 PNG/JPEG 或尺寸差异等“伪线索”。

2. **双流特征融合**  
   在预训练视觉编码器特征之外，引入 SRM 残差或 DCT 频域特征，构建“语义流 + 取证流”的双流检测框架。

3. **强调鲁棒性而非只看静态测试精度**  
   重点评估模型在 JPEG 压缩、缩放等扰动下的性能，而不是只追求原始测试集上的高准确率。

4. **轻量化实现**  
   优先采用冻结主干 + 小型融合头的方式，必要时配合 LoRA，以控制训练成本，保证项目可落地。

## 4. 数据与任务设计

正式实验计划使用 `CIFAKE` 数据集开展。该数据集总规模约 `120,000` 张图像，下载体量约 `100MB` 级别，远小于 `GenImage`，更适合课程项目在普通个人设备上完成训练和实验。为了控制项目规模，我们优先在标准训练/测试划分上完成 baseline 和改进模型，再额外构造压缩与缩放扰动测试集。除此之外，项目还引入 `HybridForensics` 作为第二 benchmark。该数据集包含 `5000 real + 5000 fake`，假图同时覆盖 `GAN` 与 `Diffusion` 两类生成范式，因此更适合作为外部 benchmark 来测试模型的跨数据集泛化能力。工程上还额外保留了 `Hemg/AI-Generated-vs-Real-Images-Datasets` 的自划分入口，可按 `8 / 1 / 1` 生成补充训练集，用于验证结论在更大训练规模下是否仍然成立。

任务设置包括：

1. **基础二分类任务**：判断图像是真实图像还是 AI 生成图像。
2. **标准测试集评估**：在 CIFAKE 官方测试集上评估模型的基本判别能力。
3. **鲁棒性测试**：评估 JPEG 压缩、缩放等后处理对检测性能的影响。
4. **外部 benchmark 测试**：在 `HybridForensics` 上直接测试训练好的模型，用于观察跨数据集泛化能力。
5. **偏差控制设置**：所有输入图像统一分辨率，并重新保存为相同 JPEG 质量，以降低格式和尺寸偏差带来的虚高结果。

为保证实验可复现性，当前工程统一把数据切分和训练的随机种子固定为 `4210`。

## 5. 方法设计

整体路线分为三步：

1. **建立 baseline**  
   第一阶段优先采用一个冻结预训练视觉编码器加线性分类头的 `CLIP ViT-B/32 + linear head` 作为主要 baseline。该 baseline 来源于 `CLIP` 系列视觉-语言预训练方法，原始方法由 `Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021` 提出；结合当前工程实现，实际采用的是 `transformers` 中的 `openai/clip-vit-base-patch32` 视觉编码器，并在其上接一个线性二分类头。选择这一 baseline 的原因是其预训练视觉语义特征较强、训练成本低，并且天然适合作为后续双流融合模型中的语义分支。  
   为了形成更完整的对照实验，项目还设置两个 CNN baseline。第一个是 `ResNet18`，来源于经典残差网络 `He et al., Deep Residual Learning for Image Recognition, CVPR 2016`，作为传统 CNN 代表；第二个是 `MobileNetV3-Small`，来源于轻量网络 `Howard et al., Searching for MobileNetV3, ICCV 2019`，作为更轻量、更弱的 CNN baseline。二者将用于和 `CLIP` 基线比较在小样本、轻量化约束和扰动测试下的性能差异。

2. **构建双流模型**  
   一条分支提取高层视觉语义特征，另一条分支提取 SRM 残差或 DCT 频域统计特征；随后通过拼接或小型 MLP 完成融合分类。第一版改进模型不追求复杂结构，重点保证复现简单、对比清晰。

3. **分析泛化与失效原因**  
   比较单流和双流模型在原始测试集、不同压缩条件以及外部 benchmark 上的表现，并讨论模型到底利用了哪些特征。当前首轮外部测试已经显示：在 `CIFAKE` 上训练的轻量 CNN baseline 迁移到 `HybridForensics` 后仅取得约 `0.59` 的 AUC，这说明模型存在明显的数据集依赖，也为后续引入底层取证特征提供了直接动机。

### 方法设计的理论支撑

本项目采用双流融合并不是凭直觉堆结构，而是有较明确的文献依据。首先，从一般方法论上看，多视角表示学习（multi-view representation learning）强调：当不同视角包含互补信息时，表示融合通常比单一视角更稳健。`Li et al., A Survey of Multi-View Representation Learning, arXiv 2016 / TKDE 2018` 从“表示对齐”和“表示融合”两个角度系统总结了这一点，为“语义流 + 取证流”的设计提供了通用理论背景。

其次，在图像取证任务中，`Zhou et al., Learning Rich Features for Image Manipulation Detection, CVPR 2018` 直接提出了 `RGB stream + noise stream` 的双流框架，其中噪声流利用 `SRM` 滤波器提取残差特征，并通过双线性池化与 RGB 特征融合。论文实验表明，该双流方法优于各自单流分支，并且对缩放和压缩具有更好的鲁棒性。这说明“高层视觉内容 + 底层噪声残差”的互补融合在取证任务中是有直接先例支持的。

再次，频域和压缩痕迹分支同样有明确依据。`Kwon et al., CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing, WACV 2021` 采用 `RGB + DCT` 双流结构联合学习压缩伪影相关特征，说明空间域与频域信息联合建模是图像取证中的有效路线。与此一致，`Luo et al., Generalizing Face Forgery Detection With High-Frequency Features, CVPR 2021` 指出许多 CNN 检测器容易过拟合到特定生成方法的颜色纹理，而高频特征能够暴露更稳定的伪造痕迹，并在跨数据集场景下带来更好的泛化表现。对本项目而言，这几篇论文共同支持一个核心判断：当 baseline 在同分布数据上表现很好、但在外部 benchmark 上显著退化时，引入 `SRM` 残差或 `DCT` 频域分支的双流融合，是一种有理论基础且可解释的改进方向。

如果时间允许，还会加入可解释性分析，如热力图或特征可视化，用于说明模型关注的是语义结构还是局部伪影。

## 6. 预期结果

我们预期：

1. 冻结式 `CLIP ViT-B/32 + linear head` baseline 在标准测试集上能取得不错结果，但在压缩、缩放和外部 benchmark 场景下会明显退化；
2. 加入底层取证特征后，模型在扰动场景和跨数据集测试中的鲁棒性会有所提升；
3. `ResNet18` 与 `MobileNetV3-Small` 这类 CNN baseline 预计在小样本和分布扰动下更容易出现性能下降，从而为后续融合方法提供清晰的对照参照；
4. 当前首轮结果已经表明，`MobileNetV3-Small` 在 `HybridForensics` 上只有约 `0.523` 的 Accuracy 和 `0.592` 的 AUC，这说明仅在 `CIFAKE` 上取得高分并不意味着具有良好的跨数据集泛化能力；
5. 经过偏差控制后的实验结果会更可信，也更能体现方法本身是否有效。

## 7. 与课程内容的关联

本项目与课程内容的联系主要体现在：

1. 深度神经网络与图像表示学习；
2. 迁移学习/预训练模型的使用；
3. 模型鲁棒性与分布扰动分析；
4. 特征融合与实验评估设计。

## 8. 可行性与实施计划

为了保证项目完整性，实施顺序计划如下：

1. 完成 `CIFAKE` 数据准备、统一预处理和 baseline 搭建；
2. 在 `HybridForensics` 上补充外部 benchmark 测试；
3. 训练并评估双流融合模型；
4. 完成压缩、鲁棒性和跨数据集实验；
5. 整理结果、撰写报告并准备展示材料。

该方案的优点是：**问题明确、技术路线清晰、具有一定创新性，并且在课程项目的时间和算力限制下可实现。**

## 9. 代表性参考文献

1. `Radford et al.`, *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
2. `He et al.`, *Deep Residual Learning for Image Recognition*, CVPR 2016.
3. `Howard et al.`, *Searching for MobileNetV3*, ICCV 2019.
4. `Li, Yang, and Zhang`, *A Survey of Multi-View Representation Learning*, arXiv 2016 / IEEE TKDE 2018.
5. `Zhou et al.`, *Learning Rich Features for Image Manipulation Detection*, CVPR 2018.
6. `Kwon et al.`, *CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing*, WACV 2021.
7. `Luo et al.`, *Generalizing Face Forgery Detection With High-Frequency Features*, CVPR 2021.