# DPA4C Scale-Aware MD

本仓库整理固定 DPA4C Nano 在 PPU 上的规模感知整步 MD 优化源码与论文数据。
维护者：Chengxiang Luo，luochengxiang@dp.tech。

源码来自论文 v1 的 `8b473d8` 冻结提交。计算源码保持原字节，保留上游署名。
主要入口是 `production/dpa4c-runtime/`；先阅读英文 README 和 `docs/BUILDING.md`。

无需 PPU 的检查：

```bash
python scripts/check_source.py
python scripts/verify_results.py
python scripts/run_cpu_checks.py
```

九题开发自测 S=3.7490935717621388。27 个计时配对是每题三次重复，不是 ABBA。
36 个物理阶段报告通过；独立计时间隔审计未运行，不能称官方认证或新版本成绩。
本次整理没有上卡、没有原生重编译、没有改 production。

原生构建依赖特定 PPU SDK、Python/PyTorch、固定模型及锁定上游源码。
大体积上游源码包适合作为 GitHub Release 附件，不能塞进普通源码提交。
模型、SDK、二进制、评测器与私有轨迹不在仓库内。

项目新增部分的代码许可证尚待维护者确定；上游许可证已附。
论文的分发许可与代码许可证是两项独立选择。
