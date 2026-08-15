# 演进织网·DeepSeek 版（EvoWeave_deepseek）

面向已有 Python 仓库的无固定业务角色、自适应任务图多智能体软件更新系统。

> 本仓库由 EvoWeave 演进而来：架构与核心机制一致，收敛为**单一模型 `deepseek-v4-flash`（按任务难度动态调整推理等级）+ 纯文本输入**；不规划多模态输入与跨供应商动态选模。

## 项目标识

| 项目项 | 正式名称 |
|---|---|
| 中文名称 | 演进织网·DeepSeek 版 |
| 英文名称 | EvoWeave_deepseek |
| GitHub 仓库 | [franklil0401/EvoWeave_deepseek](https://github.com/franklil0401/EvoWeave_deepseek) |
| Python 包名（规划） | `evoweave_ds` |

## 一句话定位

> 一个由总调度 Agent 动态生成任务图，并按任务实时装配无固定业务角色 Agent，在隔离工作区中为已有仓库生成可验证补丁的软件演进系统；统一使用 `deepseek-v4-flash`，按任务难度动态调整推理等级，仅处理文本任务。

## 与 EvoWeave 的关系

本项目复用 EvoWeave 已验证的架构与工程方法（无固定业务角色、自适应任务图、控制上下文隔离、worktree 隔离写入、补丁集成守卫、确定性验证门禁），并做两处收敛：

| 维度 | EvoWeave | EvoWeave_deepseek（第一阶段） |
|---|---|---|
| 模型策略 | 三家供应商、三档能力、动态路由与回退 | **仅 `deepseek-v4-flash` 一个模型**；按任务难度动态调整推理等级（reasoning effort：low/medium/high）；不换模型、无回退链 |
| 输入模态 | 文本 + 图片（受控摄取、视觉任务） | **仅文本**；无图片摄取、无多模态路由、无视觉任务 |

本项目**不再规划**跨供应商动态选模、模型回退链与多模态输入等扩展阶段；后续是否扩展以实际需求为准，另行评估，不预先承诺。核心领域协议的含义与 EvoWeave 保持一致。

## 当前状态

- 阶段 0 进行中：已完成项目文档基线（README、任务文档、项目结构文档），工程代码尚未引入；
- 代码基线：EvoWeave 提交 `1be0b74`（含未提交的 mypy 平台检查修复：`getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)`）；
- 本阶段交付物仅为文档，不包含任何提交。

## 文档入口

- [任务文档](任务文档.md)：目标、两个收敛改动、阶段规划与验收标准
- [项目结构文档](项目结构文档.md)：目录、文件职责与依赖边界
- [实验方案](实验方案.md)：证明方法优势的完整对比实验设计（阶段 3 执行）
- [EvoWeave 原始项目](https://github.com/franklil0401/EvoWeave)：架构、评测记录与历史证据
