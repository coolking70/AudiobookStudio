# 离线复核包与 IndexTTS 语气描述工作流

本文记录 2026-06-27 落地到主项目的两项分析后处理能力：离线强模型复核包、IndexTTS 段级语气描述生成。

## 离线强模型复核包

目标：分析流程完成后，或导入已有工作快照后，让前端用户可以把高风险段落打包发给外部强模型复核，再把强模型的裁决回写到当前项目状态。

前端入口位于分析结果区的“离线复核包”面板：

1. 完成结构化分析，或导入包含 `segments` 的任务快照。
2. 建议先运行“机器审计”，让 ⚑ / · 标记帮助复核包优先收敛到高风险段。
3. 点击“生成并下载复核包”，得到 Markdown 文件。
4. 把 Markdown 原样发给 ChatGPT、Gemini 等强模型，要求它只返回包内指定的裁决行。
5. 将裁决结果粘贴到面板文本框，或上传 `.txt / .md` 文件。
6. 点击“应用复核结果”，系统会把改判写回当前 `segments`，并刷新分析状态。

后端接口：

- `POST /api/review-packet/generate`：调用 `review_packet.build_review_packet`，输出 `content / filename / idmap / count / items`。
- `POST /api/review-packet/apply`：调用 `review_packet.apply_review_verdicts`，解析裁决行并返回更新后的 `segments` 与统计。

裁决行支持角色名、`K` 保持、`D` 待定等简短格式；回写时会结合 `idmap` 和角色清单做别名归一，避免外部模型返回轻微异体字或别名时无法命中。

## IndexTTS 语气描述生成

实验结论：把“说话人归因”和“IndexTTS 语气描述”放进同一次 LLM 输出会降低归因准确率。以 muli4 seg8 样本为例，原成熟管线参考结果为 119/143，同轮追加 `tts_style` 后降到 106/143。因此主项目采用两阶段策略：

1. 先用现有成熟管线完成 `speaker` 分析、块级复核和人工/强模型复核。
2. 在 speaker 固定后，再单独调用轻量 LLM 为每段生成短 `style / instruct`。

前端入口位于分析结果工具栏的“生成语气描述”按钮。它会：

- 使用当前 `segmentsState`，不重新判断 speaker。
- 优先使用前端当前 LLM 配置；如果配置不可用，则后端读取 `AGNES_API_KEY` 环境变量或项目根目录 `.env`。
- 通过 `POST /api/tts-style/stream` 以 SSE 返回进度。
- 完成后写回每段 `style`，并在缺失时补 `emotion="neutral"`。

生成提示词要求模型只输出 `编号 | 中文短语气描述`，描述应符合 IndexTTS 常见 instruct 用法，例如“轻柔疑惑，语速缓慢”“严肃直接，语气坚定”。后端 `tts_style_service.normalize_tts_style` 会做一次确定性清理：去掉解释性标签、英文片段和过长尾巴，并把“解释说明”“内心独白”等功能标签改写成更适合 TTS 的声音描述。

## 实测记录

当前已保留两个可复现实验脚本：

- `bench_scripts/run_agnes_tts_style_attribution_ab.py`：同轮归因 + 语气描述 A/B，结论为负。
- `bench_scripts/run_agnes_tts_style_second_pass.py`：固定 speaker 后二阶段生成语气描述，严格提示词下 143/143 有输出，人工抽查未发现明显语义错误，仅少量描述可进一步润色。

运行这些脚本需要本地设置 `AGNES_API_KEY`，输出默认写入 `bench_outputs/`，该目录作为实验产物不进入版本库。
