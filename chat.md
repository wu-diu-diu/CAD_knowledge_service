# Chat 接口设计总结

## 1. 需求总结

当前项目已经将 CAD 图像处理流程拆分为两段：

- `POST /upload-and-process`
  - 负责执行步骤 1-6
  - 完成房间识别、轮廓提取、门归属、房间几何重建、CAD 坐标转换
  - 返回 `results` 和 `session_id`
- `POST /process-all-rooms`
  - 负责执行步骤 7-8
  - 完成房间离散化、灯具布置、开关布置、布线生成
  - 返回 `lighting_results` 和 `wiring_results`

在此基础上，需要进一步实现一个持续对话的 chat 能力，使用户可以多轮指定房间、修改灯具类型、修改灯具数量、重新布线，并看到模型的推理和动作过程。

目标不是让模型重新做步骤 1-6，而是基于服务端已保存的房间中间状态，按用户要求只处理指定房间。

## 2. 总体方案

### 2.1 设计原则

- 客户端负责理解用户自然语言，生成结构化任务 JSON
- 后端负责校验 JSON、读取会话状态、执行工具链、返回结果
- 后端不负责做主要的自然语言意图解析
- 用户显式给出的参数优先级最高，不允许被默认推断覆盖

### 2.2 推荐接口

新增一个主接口：

- `POST /chat/turn`

作用：

- 接收客户端解析后的结构化意图
- 基于 `session_id` 和房间状态执行设计任务
- 返回：
  - 当前房间的设计结果
  - 布线结果
  - 本轮执行 trace（模型 thought / action / observation）

推荐辅助接口：

- `GET /chat/state`
  - 查询当前会话状态或指定房间状态
- `POST /chat/reset-room`
  - 重置指定房间的动态设计状态

## 3. 服务端状态组织方式

### 3.1 静态预处理状态

由 `upload-and-process` 生成，保存于：

- `cad_sessions/<session_id>/manifest.json`

包含：

- `session_id`
- `cad_params`
- `image_path`
- `cad_rooms`
- `room_rectangles`
- `door_assignments`

这些数据是步骤 7-8 的基础输入，也是 chat 模式的底座。

### 3.2 对话状态

建议保存于：

- `cad_sessions/<session_id>/chat/<conversation_id>/conversation.json`

包含：

- `session_id`
- `conversation_id`
- `current_room`
- `turn_count`
- `history`
- `global_preferences`

### 3.3 房间级动态设计状态

建议保存于：

- `cad_sessions/<session_id>/chat/<conversation_id>/rooms/<room_name>.json`

包含：

- `selected_lamp_type`
- `lamp_count`
- `target_lux`
- `required_flux_per_lamp_lm`
- `lamp_model`
- `placement_mode`
- `lamps`
- `switch`
- `wiring`
- `validation`
- `tool_cache`
- `execution_history`

这样可以支持：

- 针对单个房间持续修改
- 不同房间相互独立
- 多轮对话中保留已有布局

## 4. 客户端应做什么

客户端模型不是最终设计执行器，而是**意图解析器**。

客户端应承担以下职责：

1. 维护对话上下文
   - `session_id`
   - `conversation_id`
   - `current_room`
   - 上一轮显式参数
2. 将用户自然语言解析为结构化 `intent JSON`
3. 根据任务类型调用后端接口
4. 展示后端返回的 trace 和设计结果

客户端不应负责：

- 计算灯具数量
- 计算光通量
- 生成布线
- 代替后端工具链做设计推理

## 5. 客户端解析后的意图 JSON 方案

推荐格式：

```json
{
  "intent_type": "design_room",
  "room_name": "除尘室",
  "constraints": {
    "lamp_type": "防爆灯",
    "lamp_count": 4,
    "target_lux": null,
    "required_flux_per_lamp_lm": null,
    "lamp_model": null,
    "placement_mode": "rule",
    "run_wiring": true,
    "switch_count": null,
    "lamps": null,
    "switch": null
  },
  "execution": {
    "resume_existing": true,
    "start_from": "auto",
    "overwrite_existing": false
  }
}
```

### 5.1 字段说明

- `intent_type`
  - `design_room`
  - `update_room`
  - `rerun_layout`
  - `rerun_wiring`
  - `get_room_state`
  - `list_rooms`
  - `reset_room`
- `room_name`
  - 目标房间名称
- `constraints`
  - 用户显式指定的工程约束
- `execution`
  - 执行控制参数

### 5.2 参数优先级

必须遵守：

- 用户显式输入 > 当前房间已有状态 > 房间默认规范 > 系统默认值

例如：

- 用户指定 `lamp_type="防爆灯"` 时，不应再调用默认灯具匹配覆盖它
- 用户指定 `lamp_count=4` 时，不应再重新估计灯具数量

## 6. 服务端执行逻辑

收到 `/chat/turn` 请求后，建议流程如下：

1. 校验 `session_id`
2. 读取会话静态中间结果
3. 读取或创建 `conversation_id`
4. 校验 `room_name`
5. 读取该房间已有动态设计状态
6. 用 `constraints` 覆盖已有状态
7. 依据 `execution.start_from` 或自动规划规则决定从哪一步开始执行
8. 调用后端工具链
9. 保存更新后的房间状态与对话历史
10. 返回本轮结果

## 7. 自动起始阶段规划规则

建议采用半约束执行，而不是完全交给模型自由决定。

### 场景 A：只给房间名

```json
{
  "room_name": "除尘室"
}
```

执行顺序：

- requirement
- count
- flux
- model
- placement
- validation
- wiring

### 场景 B：给了灯具类型和数量

```json
{
  "room_name": "除尘室",
  "constraints": {
    "lamp_type": "防爆灯",
    "lamp_count": 4
  }
}
```

执行顺序：

- 跳过 `requirement`
- 跳过 `count`
- 从 `flux` 开始

### 场景 C：直接给了灯具位置

```json
{
  "room_name": "除尘室",
  "constraints": {
    "lamp_type": "防爆灯",
    "lamp_count": 4,
    "lamps": [[5, 4], [5, 8], [10, 4], [10, 8]]
  }
}
```

执行顺序：

- 跳过 `requirement`
- 跳过 `count`
- 跳过 `flux`
- 跳过 `model`
- 从 `validation` 开始
- 如 `run_wiring=true`，再执行 `wiring`

## 8. Trace 回传设计

为了让用户看到模型在做什么，服务端建议在响应中返回结构化 trace。

推荐格式：

```json
{
  "trace": [
    {
      "step": 1,
      "thought": "用户已明确给出灯具类型和数量，可以跳过默认匹配。",
      "action": "tool_calc_required_flux_per_lamp",
      "action_input": {
        "target_lux": 300,
        "lamp_count": 4
      },
      "observation": "单灯所需光通量约 2330 lm。"
    }
  ]
}
```

推荐只返回：

- 简短 thought
- 工具名
- 参数摘要
- 工具结果摘要

不建议把冗长内部日志原样回传给客户端。

## 9. 客户端与后端的配合流程

### 第一步：上传图片

客户端调用：

- `POST /upload-and-process`

获得：

- `session_id`
- `results`

客户端保存：

- `session_id`
- 房间列表
- 当前会话上下文

### 第二步：用户发起房间设计任务

例如用户说：

- “帮我给除尘室布置 4 个防爆灯”

客户端模型解析成 `intent JSON`，然后调用：

- `POST /chat/turn`

后端返回：

- 最新设计结果
- 布线结果
- 执行 trace

### 第三步：用户继续修改

例如：

- “把它改成 6 个灯”
- “不要防爆灯，改成双管荧光灯”
- “重新布线，灯位不要动”

客户端模型结合 `current_room` 和上一轮状态继续生成结构化意图，再调用 `/chat/turn`。

## 10. 客户端需要编写的代码模块

建议客户端按以下模块组织：

### 10.1 状态管理

- `SessionState`
  - `session_id`
  - `conversation_id`
  - `current_room`
  - `available_rooms`
  - `last_intent`

### 10.2 意图解析

- `IntentParser`
  - 输入：用户文本 + 当前上下文
  - 输出：`intent JSON`

### 10.3 HTTP API 封装

- `UploadAndProcessAsync`
- `ProcessAllRoomsAsync`
- `ChatTurnAsync`
- `GetChatStateAsync`

### 10.4 展示层

负责展示：

- 房间结果
- 灯具结果
- 布线结果
- trace
- 可视化图片路径

## 11. 推荐结论

最合理的实现方式是：

- 客户端解析自然语言为结构化意图
- 后端基于已保存的房间中间状态执行设计任务
- 后端返回结构化结果和 trace
- 客户端负责展示，而不是重复做工程计算

这样可以保证：

- 接口职责清晰
- 参数优先级明确
- 多轮对话稳定
- 工程逻辑可控
