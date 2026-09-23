# 客户端依赖版本与后续锁定

A0/G0 已安装131个包版本完全相同。A1安装时字体/绘图库fonttools从4.65.0升级到4.66.0，其余130个版本相同，Python均3.11.16；完整版本清单在client-environments/。AIPerf为同一固定InferenceX checkout的editable安装；仓库utils/aiperf工作区检查干净。包版本相同不等同于所有源码字节的事后证明，避免误称whole-tree hash。

源码搜索中，matplotlib引用位于plot/analysis，未在请求发送、调度或计时路径发现fontTools引用。A1已开始预热，保留这个差异记录并继续原实验；没有在运行过程中更换其依赖。原始吞吐/延迟来自请求记录，图表渲染不作为性能证据。

后续安装使用config/client-constraints-a0.txt固定A0实际131个版本。validate_and_pin_client.py先执行原runtime validator，再把UV_CONSTRAINT写入即将使用的runtime.env，并记录constraint文件SHA256；只约束benchmark开始前的uv安装，不改变请求、调度、warmup或持续时间。G1配置已接入该hook，但G1仍需A1分析后决定是否执行。

验证：uv --help确认UV_CONSTRAINT支持；构造runtime.env测试确认保留已有变量、追加路径和正确SHA256；在A1环境执行uv pip install --dry-run，37.24秒解析131包，计划仅重建相同固定源码的editable AIPerf，并将fonttools降回4.65.0，没有执行安装。首次dry-run用错requirements路径报File not found，修正为utils/agentic-benchmark/requirements.txt后通过；不涉及测试流量。

为减少重复下载，已将完成A0的uv-cache复制到独立的aiperf-g1-guard-completion-c80/uv-cache，未复用A0的venv、未修改A1的缓存/环境。仍走原安装流程，并核对实际安装版本。缓存准备不等于已启动G1。
