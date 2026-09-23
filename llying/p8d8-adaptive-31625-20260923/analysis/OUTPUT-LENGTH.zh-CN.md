# 输出长度核查

旧4K与A0均存在少量输出短于trace指定预算的请求。客户端output_sequence_length与服务端usage_completion_tokens一致；这不是已观察到的客户端token计数漏算。

由实际输出和导出的OSL mismatch百分比反算指定预算（四舍五入误差<1e-9 token）：旧4K actual 9,468,022 / requested 9,847,765 =96.1439%；A0 9,387,580 / 9,732,647 =96.4545%。所有非零差异分别295/9782、279/9727；AIPerf自带mismatch counter有自己的阈值，分别290、276，不混淆两者。

客户端日志确认两组scenario均注入ignore_eos=true。实际SGLang源码中该开关只使token/EOS分支不终止；stop string、vocab边界、grammar终止等仍是独立终止路径。本栈使用glm47 tool-call parser / glm45 reasoning parser。已读源码不能确定每条短输出的实际finish分支；未记录这些请求的原始finish reason，不将其擅自归因于grammar或NaN。A0 decode日志未发现NaN/越界/ERROR告警。

这是现有benchmark行为和可比性限制。G0继续同一配置，最终核对实际/指定输出比例，并同时比较完成数、真实output/total吞吐、相同实际输出长度的匹配请求。若主要变化来自更短生成，不认定为性能收益。不在A0/G0对照中途改变终止规则。
