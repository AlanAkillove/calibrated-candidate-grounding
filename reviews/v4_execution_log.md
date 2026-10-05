# V4 执行日志

2026-10-05：从指定基线建立 `v4-targeted-strengthening`。主代理负责设计与最终验收，三个执行代理均为 gpt-6-luna/max，独立明确任务启动。

初始保护审计 PASS：890 个受保护结果文件、3307 个 V3 冻结输入记录、834 个修补输入路径、205 个 V3 测试源码、25 个 publication 来源；5026 个不同文件 SHA256，16,912,239,886 字节，errors=[]。未运行模型或性能分析。原用户 `audit/` 未操作。

V_pure 固定12维实现和逐维依赖表完成，12项合成测试通过。旧V8全部经B3 winner或rank依赖query；这是依赖审计结果，不是新性能结果。practical指标工具完成，12项合成测试通过；此时没有真实端点。

P0：冻结研究问题、六项A/B主要family、coverage、AURC口径、12维特征、source-only阈值规则、外部前门槛与停止规则后，再允许源域训练和runner开发。实际冻结哈希见 `results/v4_targeted_strengthening/p0_freeze.json`。FineCops正式V4计算必须等待P1。

预算：A/B GPU预期0，CPU 4–24小时；外部如果通过门槛预计1–6 GPU小时、硬上限20小时。当前外部仍为数据审计，不计作跨域实验完成。
