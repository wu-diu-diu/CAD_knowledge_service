# ReAct 模式改进总结

## 修改时间
2026-03-07

## 修改文件
`agent/react_agent.py`

## 五点建议及实现状态

### 1. ✅ 修复循环终止逻辑

**问题**: `tool_apply_layout_edit` 没有写 `continue`，导致进入 `else` 分支意外终止

**修改内容**:
- 在 `tool_apply_layout_edit` 后添加 `continue` 语句 (line 1259)
- 增加 `VALID_ACTIONS` 集合用于明确验证 (line 1137-1148)
- 改进 else 分支，区分 unknown 和 unhandled 动作 (line 1260-1267)

**关键代码**:
```python
VALID_ACTIONS = {
    "finish", "tool_validate_layout", "tool_lookup_room_requirement",
    "tool_estimate_component_count", "tool_calc_required_flux_per_lamp",
    "tool_retrieve_lamp_model", "tool_place_components",
    "tool_read_matrix_state", "tool_generate_wiring", "tool_apply_layout_edit",
}

# ...
if action.get("action") == "tool_apply_layout_edit":
    args = action.get("args", {}) or {}
    result = self.tools.tool_apply_layout_edit(state=state, edits=args.get("edits", []))
    last_tool_output = {"tool": "tool_apply_layout_edit", "output": result}
    continue  # ← 添加的 continue
else:
    if action.get("action") not in VALID_ACTIONS:
        final_reason = f"unknown_action:{action.get('action')}"
        break
    final_reason = f"unhandled_action:{action.get('action')}"
    break
```

---

### 2. ✅ 增强提示词 (REACT_SYSTEM_PROMPT)

**问题**:
- 缺少工具输出说明
- 缺少 Few-shot 示例
- 缺少错误处理指导
- 执行策略过于绝对
- "你只能通过工具行动" 表述不清晰

**修改内容** (line 1752-1938):

1. **添加工具输出说明**:
   ```python
   工具说明与输出:
   1) tool_lookup_room_requirement
      输入: {"room_name":"可选"}
      输出: {
        "room_name": "房间名",
        "target_lux": 300,  # 目标照度
        "lamp_type": "筒灯",  # 推荐灯具类型
        "preferred_lamp_types": ["筒灯"],  # 可选灯具列表
        "constraints": {...}
      }
   ...
   ```
   为所有 10 个工具添加了详细的输入/输出说明

2. **添加 Few-shot 示例**:
   ```python
   Few-shot 示例:

   示例1: 完整布局流程
   ---
   模型输入: 空房间，面积12.5m²，名称"办公室"
   {
     "thought": "房间为办公室，需要查询照度要求和推荐灯具类型",
     "action": "tool_lookup_room_requirement",
     "args": {}
   }
   ---
   Observation: 工具返回 {"target_lux": 300, "lamp_type": "筒灯", ...}
   ...
   ```
   提供了完整的 11 步工作流示例，展示了从空布局到 finish 的全过程

3. **添加错误处理指导**:
   ```python
   错误处理:
   - 如果工具返回错误或无效结果，仔细阅读Observation中的错误信息，调整参数后重试。
   - 如果连续多次调用同一工具失败，考虑调整策略或使用其他工具。
   - 如果校验得分<70，根据suggestions调整布局。
   ```

4. **改进执行策略描述**:
   - 强调工作流程是"基于 Observation 的决策"
   - 添加"每轮只调用一个动作"的明确说明

5. **完善工具参数说明**:
   为每个工具添加了详细的输出字段说明和注释

---

### 3. ✅ 完善循环，添加 Observation 回环

**问题**: 缺少 Observation 回环，模型看不到工具的输出结果

**修改内容**:

1. **添加 `last_tool_output` 变量** (line 1151):
   ```python
   last_tool_output: Optional[Dict[str, Any]] = None
   ```

2. **修改 `_decide_action` 签名** (line 1406):
   ```python
   def _decide_action(
       self,
       state: RoomAgentState,
       view: Dict[str, Any],
       validation: Dict[str, Any],
       last_tool_output: Optional[Dict[str, Any]] = None,  # ← 新增参数
   ) -> Dict[str, Any]:
   ```

3. **每个工具执行后保存输出** (line 1175-1259):
   ```python
   if action.get("action") == "tool_lookup_room_requirement":
       args = action.get("args", {}) or {}
       result = self.tools.tool_lookup_room_requirement(
           state=state,
           room_name=args.get("room_name"),
       )
       last_tool_output = {"tool": "tool_lookup_room_requirement", "output": result}  # ← 保存输出
       continue
   ```
   为所有 9 个工具添加了输出保存逻辑

4. **传递 `last_tool_output` 给 LLM** (line 1155):
   ```python
   action = self._decide_action(state, view, validation, last_tool_output)
   ```

5. **在提示词中展示 Observation** (line 1447-1452):
   ```python
   if last_tool_output:
       prompt_parts.extend([
           "",
           "=== 上轮工具执行 (Observation) ===",
           f"工具: {last_tool_output.get('tool')}",
           f"输出: {json.dumps(last_tool_output.get('output'), ensure_ascii=False, indent=2)}",
       ])
   ```

**实现效果**:
- 模型现在可以看到上一轮工具的完整输出结果
- 形成了真正的 ReAct 循环: 状态 → Thought → Action → Observation → 下一个 Thought

---

### 4. ✅ 增加历史窗口和改进格式

**问题**:
- 历史窗口太小（8次）
- user_prompt 格式不够直观
- 只用 JSON 格式，缺少结构化信息

**修改内容**:

1. **增加历史窗口** (line 1429):
   ```python
   HISTORY_WINDOW = 16  # 从 8 增加到 16
   ```

2. **创建 `_build_user_prompt` 方法** (line 1422-1468):
   ```python
   def _build_user_prompt(
       self,
       state: RoomAgentState,
       view: Dict[str, Any],
       validation: Dict[str, Any],
       last_tool_output: Optional[Dict[str, Any]] = None,
   ) -> str:
       HISTORY_WINDOW = 16
       prompt_parts = [
           "=== 房间状态 ===",
           f"房间名: {state.room_name}, 面积: {state.area_m2:.2f}m²",
           f"已放置: 灯具{len(state.placements.get('lamps', []))}个, 开关{len(state.placements.get('switches', []))}个",
           f"已选灯具: {state.selected_lamp_type or '未选择'}",
           "",
           "=== ASCII 棋盘 ===",
           "#=障碍, .=可布置, D=门, L=灯, S=开关",
           view["ascii_board"],
           "",
           "=== 校验结果 ===",
           f"得分: {validation.get('score', 0)}/100",
           f"是否有效: {validation.get('is_valid', False)}",
           f"违规数: {len(validation.get('violations', []))}",
       ]
       ...
   ```

3. **改进提示格式**:
   - 使用分隔线（`===`）清晰划分不同部分
   - 使用中文标题增强可读性
   - 为 ASCII 棋盘添加图例说明
   - 将 JSON 输出格式化为缩进形式
   - 添加近期决策历史（包含 thought 和 action）

4. **整合到 `_decide_action`** (line 1415):
   ```python
   user_prompt = self._build_user_prompt(state, view, validation, last_tool_output)
   ```

**实现效果**:
- 提示更结构化，模型更容易理解
- 历史窗口增加，减少上下文丢失
- 信息层次清晰：状态 → 棋盘 → 校验 → Observation → 历史 → 决策

---

### 5. ✅ 完善 thought_history

**问题**:
- 只保存 `thought` 字段，没有保存 `action` 和 `args`
- 历史记录不连贯

**修改内容** (line 1161-1168):
```python
# 修改前:
state.thought_history.append({
    "thought": action.get("thought"),
})

# 修改后:
state.thought_history.append({
    "thought": action.get("thought"),
    "action": action.get("action"),
    "args": action.get("args", {}),
    "timestamp": datetime.now().isoformat(),
})
```

**实现效果**:
- 思考历史更完整，可以追溯每个决策的理由和行动
- 添加时间戳，便于调试和分析
- 在 `_build_user_prompt` 中展示 thought 和 action 的关联（line 1471-1474）

---

## 代码统计

| 项目 | 修改前 | 修改后 | 变化 |
|------|--------|--------|------|
| `run_for_room` 方法行数 | ~140 | ~160 | +20 |
| `_decide_action` 方法行数 | ~30 | ~10 | -20 (委托给 _build_user_prompt) |
| 新增方法 | 0 | 1 | `_build_user_prompt` |
| `REACT_SYSTEM_PROMPT` 行数 | ~57 | ~187 | +130 |
| 历史窗口大小 | 8 | 16 | +100% |

---

## 关键改进点

### 1. 真正的 ReAct 实现
**修改前**: 只是一个工具调用循环，模型看不到工具输出
**修改后**: 完整的 Observation 回环，模型可以基于工具输出进行推理

### 2. 更强大的提示词
**修改前**: 只有工具输入说明，缺少输出描述和示例
**修改后**: 完整的工具说明、Few-shot 示例、错误处理指导

### 3. 更好的上下文管理
**修改前**: 小历史窗口、JSON 格式不直观
**修改后**: 大历史窗口、结构化提示、清晰的信息分层

### 4. 更完善的调试信息
**修改前**: thought_history 只有 thought
**修改后**: 包含 action、args、timestamp，便于追踪

---

## 潜在影响和注意事项

### 正面影响
1. ✅ **模型决策能力提升**: 现在可以看到工具输出，可以做出更明智的决策
2. ✅ **调试更容易**: 更完整的日志和结构化提示
3. ✅ **稳定性提升**: 修复了 `tool_apply_layout_edit` 的 bug，添加了 VALID_ACTIONS 验证

### 注意事项
1. ⚠️ **token 消耗增加**: 提示词从 ~57 行增加到 ~187 行，历史窗口翻倍
2. ⚠️ **响应时间增加**: 更大的 prompt 可能导致 LLM 响应变慢
3. ⚠️ **成本增加**: 更多的 token 消耗意味着更高的 API 调用成本

### 建议优化方向
1. **考虑使用长上下文模型**: 如 Qwen-Long、GPT-4-Turbo-128k 等
2. **动态历史窗口**: 根据任务复杂度调整 `HISTORY_WINDOW`
3. **提示词压缩**: 对于 Few-shot 示例，可以考虑只保留关键步骤

---

## 测试建议

### 单元测试
```python
def test_tool_output_observation():
    agent = ReActLightingAgent(provider="qwen")
    state = RoomAgentState(...)
    last_output = {"tool": "tool_lookup_room_requirement", "output": {...}}
    # 验证 last_output 被正确传递给 _decide_action
    # 验证 _build_user_prompt 正确展示 Observation
```

### 集成测试
```python
def test_full_react_cycle():
    # 运行完整的 run_for_room 流程
    # 验证: 每轮工具执行后，last_tool_output 被更新
    # 验证: _decide_action 收到正确的 last_tool_output
    # 验证: 最终生成的提示词包含完整的 Observation 历史
```

### 性能测试
```python
def test_token_usage():
    # 对比修改前后的 token 消耗
    # 对比修改前后的响应时间
```

---

## 验证结果

✅ Python 语法检查通过
✅ AST 解析成功
✅ 所有关键修改已验证
- `VALID_ACTIONS` 已定义
- `last_tool_output` 已正确使用（9 处）
- `thought_history` 已完善
- `_build_user_prompt` 方法已创建
- `HISTORY_WINDOW` 已设置为 16
- `REACT_SYSTEM_PROMPT` 包含：
  - ✅ 工具说明与输出
  - ✅ Few-shot 示例
  - ✅ 错误处理指导
- `tool_apply_layout_edit` 后有 `continue`
- `last_tool_output` 传递给 `_decide_action`

---

## 后续工作

1. **性能优化**: 监控 token 消耗和响应时间
2. **日志增强**: 在 `AgentRunLogger` 中记录提示词和响应
3. **配置化**: 将 `HISTORY_WINDOW` 和提示词模板提取到配置文件
4. **测试覆盖**: 添加单元测试和集成测试
5. **监控**: 添加 Prometheus/Metrics 监控决策质量
