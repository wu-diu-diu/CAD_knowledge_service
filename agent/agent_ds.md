# Agent 系统改进方案 (agent_ds.md)

> 本文档基于对 `agent/` 目录下 ReAct 电气照明 Agent 代码的全面审查，列出面向生产级 Agent 系统的架构缺失、领域能力不足和工程质量问题，并给出优先级排序的改进建议。

---

## 一、当前架构概览

| 模块 | 职责 | 成熟度 |
|------|------|--------|
| `react_agent.py` | ReAct 循环 (Thought→Action→Observation) | 基础可用 |
| `mini_model.py` | Anthropic 兼容接口的备选 Agent | 基础可用 |
| `tools.py` | 15 个工具定义 + 系统 Prompt + 灯具目录 + 照度标准 | 较完善 |
| `state.py` | 房间状态 (`RoomAgentState`) + 管理器 + 历史记录 + ASCII 棋盘 | 合理 |
| `chat.py` / `run.py` | 交互式 CLI 会话 / 批处理管线 | 基础可用 |
| `image_process.py` | 建筑平面图 → 离散网格 (0/1/2) 预处理 | 可用 |
| `logger.py` | 彩色终端日志 + 文件日志 + event_sink | 可用 |

**关键指标:**
- 工具数量: 15 (查询需求/估算数量/光通量计算/型号检索/元件布置/校验/布线/位置微调/矩阵读取/需求解析/规范检索/设计汇总/合规检查/问题诊断/报告生成)
- 支持 LLM Provider: Qwen (DashScope API) / DeepSeek / MiniMax (Anthropic API)
- 设计阶段划分: 需求解析 → 设计 → 检查与修正 → 收尾

---

## 二、核心架构缺失 (按严重程度)

### 2.1 无 Planning / Reflection 模块 [严重]

**现状:** ReAct 循环仅在每轮把历史文本喂给 LLM，LLM 直接从当前状态跳到下一步决策。

**缺失:**
- **全局计划 (Plan):** Agent 启动时应先生成高层步骤序列和每步预期结果，再按计划推进而非"走一步看一步"
- **结构化自省 (Reflection):** 每步执行后应有: "当前结果是否符合计划预期? 偏差在哪? 下一步策略需调整吗?"
- **计划重规划 (Re-plan):** 校验失败或工具结果异常时，应重新评估整体计划而非仅局部修补

**改进方案:**
```python
class AgentPlan:
    phases: List[PlanPhase]          # 设计阶段列表
    current_phase_index: int
    expected_outcomes: Dict[str, Any]  # 每步预期结果

def reflect(self, plan, observation) -> Reflection:
    # 结构化自省: 偏差分析 + 置信度 + 策略调整建议
```

---

### 2.2 无错误恢复 / 自愈机制 [严重]

**现状:**
- LLM 返回 JSON 解析失败 → 直接 `finish` (`react_agent.py:412`)，无重试
- 工具执行异常被 `except Exception` 吞掉，无降级策略
- 无指数退避、无 fallback 工具链

**改进方案:**
1. LLM 输出校验失败时，追加错误提示作为下一条消息让 LLM 修正，最多重试 2 次
2. 工具调用失败时，捕获具体异常类型并映射为可操作的错误消息
3. 实现工具级 fallback 链: 如 `tool_query_design_standard` 失败 → 降级为规则库查表
4. 定义电路断路器: 连续 N 步无进展 → 触发人工干预或安全中止

---

### 2.3 无 Human-in-the-Loop (人机协作) [严重]

**现状:** Agent 执行全流程无任何暂停、确认或询问机制。

**缺失:**
- **审批门 (Approval Gate):** 在关键决策点 (如元件布置后、布线前) 暂停，展示中间结果给用户确认
- **歧义询问:** 当用户需求与规范/最佳实践冲突时 (如"放 6 个灯"但该房间最佳是 4 个)，无法主动向用户澄清
- **中断/取消:** Agent 启动后无法中途停止正在运行的设计任务

**改进方案:**
```python
class ApprovalGate:
    stage: str                      # "placement_done" / "wiring_ready"
    summary: Dict[str, Any]         # 当前设计摘要
    requires_approval: bool
    timeout_seconds: int            # 超时后使用默认策略

def request_approval(gate: ApprovalGate) -> bool:
    # 阻塞等待用户确认或超时
```

---

### 2.4 无状态回滚 / 版本管理 [严重]

**现状:** `tool_apply_layout_edit` 执行后无 undo；Agent 走错一步只能重置整个房间。

**缺失:**
- 状态快照 (Snapshot): 关键步骤前后自动保存状态
- 回滚 (Rollback): 恢复到任意之前的快照
- 分支设计: 从同一状态分叉生成多个候选方案

**改进方案:**
```python
class RoomAgentState:
    def snapshot(self) -> str:
        # 返回 snapshot_id，存储当前 placements/tool_cache/lamp_plan 的深拷贝
    def restore(self, snapshot_id: str) -> None:
        # 恢复到指定快照
    def list_snapshots(self) -> List[SnapshotMeta]:
        # 列出所有快照及时间戳/描述
```

---

### 2.5 Context Window 管理缺失 [高]

**现状:** 硬编码为最近 12 条工具结果 + 8 条思考 (`react_agent.py:436-501`)，超出直接截断。

**缺失:**
- Token 预算管理: 不追踪当前上下文占用了多少 token
- 智能摘要: 旧历史应压缩为结构化摘要而非直接丢弃
- 分层记忆: 近期工作记忆 vs 长期知识 vs 会话历史

**改进方案:**
1. 引入 `tiktoken` 或 `transformers` 进行 token 计数
2. 当接近上下文窗口上限时，调用 LLM 对旧历史做结构化摘要
3. 分层存储: 工作记忆 (当前设计) → 会话记忆 (多房间) → 持久记忆 (用户偏好/历史设计)

---

### 2.6 无并行工具执行 [高]

**现状:** 15 个工具全部串行。以下工具对之间无依赖，可并行:

| 工具 A | 工具 B | 独立性 |
|--------|--------|--------|
| `tool_lookup_room_requirement` | `tool_query_design_standard` | 完全独立 |
| `tool_estimate_component_count` | `tool_calc_required_flux_per_lamp` | 当 lamp_count 由用户指定时可并行 |
| `tool_validate_layout` | `tool_diagnose_layout_issue` | 可并行(诊断不依赖校验结果) |

**改进方案:**
LLM 一次可返回多个 action (batch)，由 Agent 调度执行无依赖的工具并行调用。

---

### 2.7 `run.py` 存在 Bug [高]

```python
# run.py:226-227 — cad_params 未定义
def run_agent_pipeline(
    image_path: Path,
    output_dir: Optional[Path] = None,
    agent_type: str = "react",
    provider: str = "qwen",
    model_name: Optional[str] = None,
    max_steps: int = 8,
) -> Dict[str, Any]:
    if cad_params is None:          # NameError: name 'cad_params' is not defined
        cad_params = dict(DEFAULT_CAD_PARAMS)
```

**修复:** 参数列表增加 `cad_params: Optional[Dict[str, float]] = None`。

---

### 2.8 无流式输出 [高]

**现状:** LLM 调用全部使用非流式 API (`chat.completions.create`)，用户看不到 Agent 实时思考过程。

**改进方案:**
使用 `stream=True`，通过 `event_sink` 将 thought/action 实时推送到前端，同时将 ASCII 棋盘用 diff 形式展示变化。

---

### 2.9 无结构化可观测性 [中]

**现状:** 日志仅为终端彩色文本 + 文件追加。无结构化指标。

**缺失:**
- Token 用量 / 成本统计
- LLM 延迟 vs 工具执行延迟的分段度量
- 成功率 / 合规率 / 校验通过率趋势
- 工具调用频率分布热力图
- 每个 Agent run 的完整 trace (类似 OpenTelemetry span)

**改进方案:**
`event_sink` 已有但未被充分利用。应扩展为结构化事件流，包含 `run_id`, `step_number`, `latency_ms`, `token_count` 等字段。

---

## 三、电气照明领域能力缺失

### 3.1 眩光 / 统一眩光值 (UGR) [严重]
GB50034 要求办公室等场所 UGR ≤ 19。目前完全未涉及:
- 灯具目录无 UGR 参数
- 无眩光计算工具
- 所选灯具即使照度达标，UGR 可能超标

### 3.2 显色指数 (CRI/Ra) [严重]
GB50034 要求办公室 Ra ≥ 80，精密工作 Ra ≥ 90。目前灯具目录无 CRI 字段。

### 3.3 应急照明 [高]
完全缺失:
- 无应急灯 / 疏散指示灯选型和布置
- 无独立应急回路设计
- 无备用电源容量计算

### 3.4 电气负载计算 [高]
当前只计算了总功率，缺少:
- 回路电流计算 (I = P / (U × cosφ))
- 断路器额定电流选型
- 电压降校验 (线路末端电压 ≥ 额定电压 × 95%)
- 三相平衡分配
- 电缆截面选型

### 3.5 LPD (照明功率密度) 精确校核 [中]
`tool_check_standard_compliance` 中 LPD 限值硬编码 (`tools.py:973`):
```python
lpd_limit = 9.0 if target_lux >= 300 else 7.0
```
实际 GB50034 分房间类型有精确限值表 (如办公室 ≤ 8W/m²，会议室 ≤ 9W/m²)，应结构化存储。

### 3.6 日光利用 / 窗户影响 [中]
矩阵中无窗户编码，无法:
- 识别日光区域
- 做日光补偿照度计算
- 日光区独立控制分组

### 3.7 灯光分区 / 分组控制 [中]
所有灯具在同一回路，无:
- 行/列分组独立开关
- 调光回路
- 场景控制 (全亮/半亮/夜灯)

### 3.8 多房间全局优化 [低]
每房间独立设计，无法:
- 优化全局配电箱位置
- 最小化管线总长度
- 全局 LPD 校核

---

## 四、工程质量问题

### 4.1 错误处理

| 问题 | 位置 | 影响 |
|------|------|------|
| `except Exception` 吞所有错误 | `tools.py:881`, `react_agent.py:409`, 多处 | 调试困难，无法区分可恢复错误与致命错误 |
| `except ImportError` 双路径导入 | `chat.py:23-34`, `run.py:22-33` | 代码异味，应统一包结构 |

### 4.2 重复代码

| 重复内容 | 出现位置 |
|---------|---------|
| Agent 构建逻辑 (`_build_agent`) | `chat.py:53-71` / `run.py:57-89` |
| 房间状态创建 (`_build_room_states`) | `chat.py:74-86` / `run.py:94-103` |
| 布线 payload 构建与 `process_room_wiring_layout` 调用 | `chat.py:321-348` / `run.py:166-185` |
| `_build_strategy_summary` | `react_agent.py:513-518` / `mini_model.py:316-320` |

**改进:** 抽取公共模块 `agent/pipeline.py` 或 `agent/shared.py`。

### 4.3 无 Agent 基类

旧的具体 Agent 类有相同的方法签名 (`run_for_room`, `_build_strategy_summary`, `list_tools`) 但无基类或 Protocol 约束。

```python
# 应添加
from typing import Protocol, Dict, Any

class LightingAgent(Protocol):
    def run_for_room(self, state, max_steps, user_goal, reset_layout) -> Dict[str, Any]: ...
    def list_tools(self) -> List[Dict[str, Any]]: ...
```

### 4.4 状态管理

- `state.tool_cache` 是 `Dict[str, Any]`，key 靠字符串约定而非枚举，拼写错误不会报错
- `placements` 用 `List[List[int]]` 无坐标验证，可能混入非法值
- 无状态变更的 immutable 记录 (event sourcing)

### 4.5 魔法数字

| 数值 | 含义 | 位置 |
|------|------|------|
| `uf=0.6, mf=0.8` | 利用系数/维护系数默认值 | `tools.py:241`, `react_agent.py:186` |
| `max_steps=6` | ReAct 默认步数 | `react_agent.py:58` |
| `history_window=12` | 工具历史窗口 | `react_agent.py:436` |
| `min_spacing_m=2.0, max_spacing_m=3.0` | 灯具间距默认值 | `tools.py:162` |
| `fill_ratio >= 0.90` | 判定规则房间的阈值 | `tools.py:179` |

**改进:** 全部提取到 `config.py` 或环境变量。

---

## 五、其他缺失功能

### 5.1 设计替代方案生成
同一房间应可生成 2-3 个候选方案 (如: 经济型/标准型/高配型)，由用户或评分函数选择最优。

### 5.2 长期记忆 / 用户偏好学习
- 同用户多次会话应复用偏好 (如总是选暖色温、某厂家灯具)
- 相似房间应复用设计模板

### 5.3 多模态输入
当前只处理 PNG 平面图。应支持:
- DWG/DXF 直接解析
- 手绘草图识别
- BIM IFC 导入

### 5.4 API 服务化
当前只有 CLI (`chat.py` 的 `input()` 循环 / `run.py` 的 `main()`)。需要封装为 FastAPI endpoint，支持异步请求和 WebSocket 流式推送。

### 5.5 测试覆盖
`AGENTS.md` 确认目前无自动化测试。需至少覆盖:
- 工具函数的单元测试 (照度公式、坐标转换、布线算法)
- ReAct 循环的集成测试 (mock LLM 响应)
- 校验规则的边界测试

---

## 六、改进优先级排序

```
P0 (阻塞上线):
├── 修复 run.py NameError bug
├── 实现 Planning + Reflection 模块
└── 实现状态快照/回滚

P1 (核心体验):
├── Human-in-the-Loop 审批点
├── 流式 LLM 输出
├── 错误恢复/重试机制
├── 补全 UGR/CRI/应急照明 领域能力
└── Context Window 智能管理

P2 (工程质量):
├── Agent 基类抽象
├── 消除 chat.py/run.py 重复代码
├── 结构化可观测性 (token 计数/延迟/成功率)
├── LLM 输出校验层 (解析失败重试)
├── 工具级 fallback 链
└── 并行工具执行

P3 (增强):
├── 设计替代方案生成
├── LPD 精确校核表
├── 电气负载计算
├── 灯光分区/分组
├── 日光利用分析
├── 长期记忆/用户偏好
├── API 服务化 (FastAPI + WebSocket)
└── 自动化测试套件
```

---

## 七、建议的目录结构

```
agent/
├── __init__.py
├── base.py              # LightingAgent Protocol / 基类
├── config.py            # 集中配置 (魔法数字、环境变量)
├── state.py             # RoomAgentState (增加 snapshot/restore)
├── plan.py              # Planning + Reflection 模块
├── reflect.py           # 结构化自省
├── tools.py             # LightingTools + 工具定义
├── catalog.py           # 灯具目录 (从 tools.py 拆分)
├── standards.py         # 标准/规范数据 (LPD 表、UGR 限值)
├── react_agent.py       # ReAct 循环实现
├── mini_model.py        # MiniMax/Anthropic Agent
├── pipeline.py          # 公共管线 (从 chat.py/run.py 抽取)
├── chat.py              # 交互式 CLI
├── run.py               # 批处理管线入口
├── image_process.py     # 图像预处理
├── logger.py            # 日志 + 结构化事件
├── memory.py            # 会话/长期记忆管理
├── guardrails.py        # 输入安全/输出校验
├── observability.py     # Token 计数、延迟追踪
├── approval.py          # Human-in-the-Loop 审批门
├── agent_ds.md          # 本文档
└── tests/               # 测试套件
    ├── test_tools.py
    ├── test_state.py
    ├── test_react_agent.py
    └── test_pipeline.py
```
