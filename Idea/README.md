# DDA4210 项目材料索引

当前 `Idea` 目录中的文档已经对应到项目推进的不同阶段，可按下面顺序直接使用：

1. `ver1.md`
   项目的总述版提案，适合给老师或组员快速说明题目、意义、创新点和整体路线。

2. `data_protocol.md`
   数据集子集选择、采样规模、偏差控制、任务定义和评价指标。

3. `baseline_plan.md`
   baseline 的具体选型、训练配置和需要记录的失效现象。

4. `fusion_model_plan.md`
   双流融合模型的结构、最小对比实验和建议的 ablation。

5. `report_and_presentation.md`
   最终报告和 13 分钟答辩所需的核心图表、页数安排和常见答辩问题。

6. `hybridforensics_note.md`
   第二 benchmark `HybridForensics` 的数据说明、首轮外部测试结果和可直接用于报告的文字表述。

7. `../data/hemg_processed/README.md`
   `Hemg` 数据集的 `8/1/1` 切分说明，默认随机种子固定为 `4210`。

## 建议使用顺序

如果你现在准备正式开工，建议按下面顺序推进：

1. 先读 `ver1.md`，确认题目和创新点不再改动
2. 再按 `data_protocol.md` 固定数据口径
3. 按 `baseline_plan.md` 先跑第一版结果
4. 再根据 `fusion_model_plan.md` 做改进模型
5. 最后按照 `report_and_presentation.md` 整理汇报材料

当前项目文档默认把复现实验的随机种子统一记为 `4210`。

## 当前项目的一句话版本

在小规模可训练数据集上，融合预训练视觉语义特征与底层取证特征，提升 AI 图像检测在压缩和缩放扰动下的鲁棒性。
