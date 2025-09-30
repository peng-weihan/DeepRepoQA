## Code-SWE-Search

### 作用
通过SWE-Search

### Environment Setup

Before running the evaluation, you'll need:

1. At least one LLM provider API key (e.g., OpenAI, Anthropic, etc.)
2. A Voyage AI API key from [voyageai.com](https://voyageai.com) to use the pre-embedded vector stores for SWE-Bench instances.

注意：
可以在examples/moatless中找到moatless相关调用的例子。
调用前需要在.env配置API，必须得配置一个是模型的API和一个是voyage AI的API用于embedding
tmp目录下面是repository和index_store，前者是git下来的有问题的库，后者是有问题的库的embedding
用jupyter notebook一个一个运行，可以观察输出结果
在examples.py中可以跑一次query查看结果

### 核心组件

#### Manager Agent
- 负责管理各个节点操作，包括MERGE、ADD、EDIT等
- 作为中央控制器，协调其他Agent的工作流程
- 根据问题和观察结果决定下一步行动

#### Plan Search Agent
- 根据问题给出观察结果
- 负责执行代码搜索和理解操作
- 使用各种搜索工具如FindClass、FindFunction、FindCodeSnippet和SemanticSearch等
- 通过ViewCode深入了解代码细节

#### Reflection Agent
- 根据解决轨迹总结经验
- 分析执行历史，评估当前回答的质量
- 决定是否继续搜索更多信息或完成回答

### 系统改动

#### 状态管理变更
- 删除所有Edit的状态，增加Search状态
- 将"issue的解决"迁移到"question"的回答这一issue
- 将Reflection Agent的判断条件改为LLM判断准确回答问题

#### 框架转变
- 将代码修改框架改成代码问答框架
- 保持各个辅助Agent不变
- 更改所有Problem Solving相关Prompt为Question Answering
- 让SWE-Search的核心Agent专注于回答的准确性和全面性

#### 工作流程
1. 理解问题：分析问题，确定需要查找的代码部分
2. 定位相关代码：使用搜索功能找到相关代码
3. 收集完整信息：查看相关代码，深入理解
4. 分析并制定答案：基于收集的信息形成完整准确的回答
5. 完成任务：当收集了足够信息后，提供全面的回答

## 代码问答框架实现细节

### Actions设计
代码问答框架中的操作主要包含以下几类：

#### 搜索操作
- `FindClass`: 通过类名搜索类定义
- `FindFunction`: 通过函数名搜索函数定义
- `FindCodeSnippet`: 搜索特定代码模式或文本
- `SemanticSearch`: 通过语义和自然语言描述搜索代码
- `FindCalledObject`: 搜索当前代码中引用但尚未找到实现的对象

#### 查看操作
- `ViewCode`: 查看特定文件或代码段
- `ListFiles`: 列出目录中的文件

#### 控制操作
- `Finish`: 完成任务，提供最终答案
- `Reject`: 拒绝当前操作路径

### Node节点结构
系统使用树状结构来管理问答过程：

- 每个Node代表代码问答流程中的一个状态
- 包含以下关键属性：
  - `action_steps`: 在节点上执行的操作序列
  - `file_context`: 文件上下文状态
  - `user_message`: 用户问题
  - `assistant_message`: 助手回答
  - `children`: 子节点列表
  - `terminal`: 表示是否为终止节点
  - `reward`: 节点的奖励值

### 消息历史管理
系统提供多种消息历史类型：

- `MESSAGES`: 标准消息格式，包含用户和助手消息以及工具调用
- `REACT`: ReAct格式，适用于思考-行动-观察循环
- `SUMMARY`: 总结格式，提供历史操作的摘要
- `MESSAGES_COMPACT`: 紧凑消息格式，减少重复信息

### 代码理解流程
1. 使用`CodeQAAgent`启动问答流程
2. 基于问题确定搜索策略
3. 使用搜索操作查找相关代码
4. 通过`ViewCode`深入理解代码
5. 使用`Finish`操作提供最终答案

## LLM设置与使用

### 模型选择
系统支持多种LLM提供商：
- OpenAI (GPT-4, GPT-3.5等)
- Anthropic (Claude)
- 其他兼容API的模型

### 响应格式
系统支持不同的LLM响应格式：
- `TOOLS`: 工具调用格式，适用于支持函数调用的模型
- `JSON`: 标准JSON格式
- `REACT`: ReAct格式的文本输出

### 提示词设计
- 系统提示词(`QA_AGENT_ROLE`): 定义了代码问答助手的角色
- 工作流提示词: 定义了执行代码问答任务的步骤
- 行动准则: 指导模型如何使用可用的工具和执行操作
- 每个动作类型有对应的提示词和少量示例

### 向量搜索
- 使用Voyage AI进行代码库的向量嵌入
- 支持语义搜索和相似度匹配
- 缓存索引以提高性能

### 使用技巧
- 确保提供足够具体的问题
- 问题应涉及代码库的具体方面
- 对于复杂问题，可以分解为多个步骤
- 利用系统中的FeedbackData机制进行反馈优化

## 从代码修改框架到代码问答框架的转变

### 核心改动
1. **目标转变**
   - 原框架: 定位并修复代码中的问题
   - 新框架: 理解代码并回答相关问题

2. **状态管理**
   - 删除了所有与编辑相关的状态
   - 增加了专注于搜索和理解的状态
   - 将问题解决流程转化为问题回答流程

3. **Agent角色调整**
   - Manager Agent保持协调角色不变
   - Plan Search Agent从寻找修复点变为寻找回答所需信息
   - Reflection Agent从评估修复质量变为评估回答准确性

4. **提示词优化**
   - 将所有Problem Solving相关提示词改为Question Answering
   - 优化了代码理解和知识收集相关的提示
   - 强调了回答的准确性和全面性

5. **评价指标**
   - 原框架: 修复是否解决了问题
   - 新框架: 回答是否有Ground Truth支持、准确、全面

### 保留的组件
- 代码搜索和理解工具
- 节点状态管理系统
- 向量搜索和嵌入机制
- 消息历史生成器

这种转变使得SWE-Search从代码修复工具转变为代码理解和问答工具，更加专注于帮助开发者理解复杂代码库的细节和架构。

### TODO List
- 记录Search过程中的路径，转化为子问题补充信息库
- Prompt优化
- 增量更新