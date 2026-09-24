# R1+R4无HiCache smoke结果

2026-09-24 18:26:54开始服务启动，18:37:29 smoke完成，约10分35秒。R1=completion，R4=on，R2/R3=off，P/D HiCache均关，有效chunk均4K。

- 8条真实P/D配对请求通过；stream guard smoke确认R1 separated=true。
- C80功能请求80/80成功，所有P/D rank均被覆盖，room和输入长度配对正确。
- 真实AIPerf客户端24/24成功并生成导出结果；客户端环境检查、依赖安装、发压、解析、结果导出链路跑通。
- 32条自然输入探针（max_tokens1/16、skip_special_tokens开/关、stream/unary）均HTTP成功且有可见内容。
- 146次P候选决策全部demand_known，所选候选均为R4最小cost、demand_suggested=true；46次与legacy选择不同。41条stream P完成日志确认separated=true。unary路径不产生同样的stream完成日志，不能以41作为全部完成请求数。

受控强制<think>（token154841，max_tokens1、skip_special_tokens=false）：chat返回HTTP200，completion_tokens=1、reasoning_tokens=1，但content为空，reasoning_content=null；实际复现了导致AIPerf InvalidInferenceResultError的响应形态。native /generate探针在Router返回404，该接口不暴露；没有把它算作自然探针或smoke成功请求。旧3条具体token仍不可追溯，此结果证明相同镜像/模型存在该解析机制，而不是证明旧3条token身份。

服务随即按计划停止；两个watchdog正常退出；正式配置启动前再次确认16张卡约0.096%占用。没有重启GPU驱动或机器。正式试验于18:40:51开始加载模型，恢复P HiCache，关闭逐候选日志，沿用C80/4K。
