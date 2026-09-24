# R1+R4离线复核与执行配置

R1在P响应体完成后释放legacy P block记账；R4按P请求的有效输入token计入ledger，并在P响应体完成后释放。两者由ActiveGuard.detach_prefill同时转移到P任务，D记录不转移。streaming路径P任务独立于客户端连接，取消D不会立即取消仍在工作的P；unary启用R1/R4时主动读取P响应体，避免等待D才释放。提前返回、失败与未派出的pick由RAII回收。选中+预约在同一锁中完成，避免并发请求看到同一个未计入的负载。

R4 score=现有在途有效输入tokens + max(本次输入tokens - prefix_hit_blocks*block_size,0)。R3关闭，使用原cache目录；R4正常路径不再以P overlap weight=20直接乘命中项，weight仍用于原策略回退。R4估计的是尚未完成请求的预约工作总量，不是随chunk完成连续递减的剩余GPU时间，也没有完整建模attention随上下文长度增长的成本。对实验结果按此边界解释。

本轮：R1=completion、R4=on、R2/R3=off；C80；TP8/DP8各8卡；P/D launcher chunk32768对应有效4096；mem .85；同基线镜像与GLM模型；P HiCache ratio1.5 write_through/kernel/page_first，D关；MTP steps5/topk1/draft6/simulated acceptance3.61；种子P823508857/D19197414。正式884条预热、3600秒窗口，逐候选日志关闭。

短时smoke：关闭P/D HiCache，保持R1+R4；真实请求与stream检查、C80功能请求、24条AIPerf小请求及32条原始响应探针。smoke不是性能对照，候选诊断日志打开。先确认模型与客户端链路，再释放无HiCache服务并启动正式配置。辅助型号/内存采集失败只报告，意外脚本失败保留服务供诊断，不自动清理健康引擎；allocation确认回收/抢占时清理当前实验。

3条空内容错误：原始记录均为warmup max_tokens=1。P/D room一致、输出token计数均1，D generation时间0，即使用P阶段首token完成。AIPerf的content_responses过滤data为空的回复，OpenAI chat解析器识别content/reasoning/tool_calls；usage-only或被过滤的单个特殊token会触发InvalidInferenceResultError。旧证据没有原始SSE和token ID，无法确定三条的具体token或区分过滤发生在哪一层；不把假设当结论。下一轮smoke保存stream/unary、max_tokens1/16、skip_special_tokens true/false的原始响应，正式客户端仍保留原错误统计，绝不将空内容误记为成功。

离线reasoning parser受控验证：从部署版本标识e7f7447333获取上游源码，仅加载StreamingParseResult/BaseReasoningFormatDetector/Glm45Detector原样类，对单独<think>作stream与unary解析均得到空content和空reasoning。tokenizer.json中<think>=154841，special=false，说明skip_special_tokens=False不保证保留reasoning分隔符。证据在review/parser-controlled-cases.json，实际镜像源码仍需在节点到位后核对。这是已复现的可能机制，三条历史请求的输出token身份仍未知。smoke追加强制154841单token的native /generate与chat接口对照。

客户端依赖准备移到模型加载前，使用独立环境和持久化Python解释器路径；正式发压阶段若该环境可用则直接复用。smoke与正式环境独立，不覆盖上一轮R2环境。硬件元信息和自动KV容量漂移记录而不按任意百分比触发退出；实际chunk、模型、策略等实验条件仍核对。

受控实机chat响应已送入同版本AIPerf ChatEndpoint.extract_chat_response_data：HTTP200/usage=1的空message返回None；填入Hello的对照返回有效data。链路复现见review/aiperf-empty-response-reproduction.json。同来源A0/G0的3条warmup均有1 token可见输出，但输入长度有0–2 token差异；这些对照不能追溯R2旧请求的具体token。

日志过滤单测已隔离到子进程，避免全局tracing callsite缓存受并行单测影响；默认并行运行276项全部通过。仅测试代码修改，生产Router二进制仍为原SHA256。
