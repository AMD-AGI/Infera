## Brief

在infera repo中的agent_sys子系统下的examples下，添加llm_e2e_performance_optimization task_package, 增加端到端拆分后的五个任务，并以glm5.3 flash为样例，在mi355机器上将每个任务独立跑通。

## Category

implemenation & experiment

## Goal

1. 在agent_sys/examples/llm_e2e_performance_optimization/为以下声明的每个步骤规定：
    1. handoff
    2. validator
    3. agent
    4. task
    5. 单独调试用的single task main
2. 使用agent_sys跑通单个task.
    1. 稳定可复现。
    2. handoff较为标准。

本轮任务重点是填充以下几个步骤，保证每个步骤单独能运行成功交付，最后串起来可运行，步骤内某算法的优劣和质量不予过多关注，主要关注handoff标准化程度、validator初步建设、task可稳定执行。

1. llm_e2e_performance_optimization 按照html规定，共分为6个阶段
    1. e2e运行
        1. 输入简单的用户指令，拿到可运行的交付脚本，格式可参考experiment-result-packup、也可进一步优化
    2. profiling拿到性能数据
        1. 执行不同的bench工具得出的profile结果
            1. 目前请包括fix_len测试\Optimus-AgenticBench: branch fix/realistic-profile-session-driver https://github.com/AMD-AGI/Optimus-AgenticBench/pull/6  、aiperf测试
                1. 首轮请集成fix_len测试和aiperf测试
            2. 要求交付profile标准产物：json文件夹，命令行分析结果，和其他工具相关标准结果：如megpie的csv文件。
    3. 分析性能数据，产生待优化算子列表，并根据列表提供完整的workset, 本轮只做标准算子：如SIKL中的定义格式
        1. 待优化算子列表的产生算法暂时不知重点，存在即可
        2. workset包括一起测试、接入、优化算子所需的运行配置和资料，至少包括：
            1. 实验环境：docker image、GPU硬件、rocm版本，其他任何在docker中变更/升级/单独安装的软件版版本
            2. sglang相关接入点的reference和说明
            3. 截取算子本身
            4. pytorch naive实现
            5. 算子运行代码：可以一键运行正确性测试、性能测试
            6. 正确性 test case 3个以上
            7. 性能测试结果：5次加权平均，每次运行loop 10次以上取平均
            8. 截取时profile报告中性能。
            9. 具体算子定义格式，本版本可参考https://github.com/AMD-AGI/SIKL ，请联系xiaobo或huangzhen
    4. kernel optimization
        1. 接入kernel forge (嵌套ai agent backend，对其提供足量kernel forge使用说明，让其嵌套调用)
            1. 输入包含workset的完成
            2. 产出标准的handoff，包括正确性包含、性能报告、优化总结
    5. 接入
        1. 拿到kernel optimization的交付物，做端到端正确性测试和性能验证，包括
            1. 服务启动的简单验证：长短文本/needle (未来还有tool call, 本轮不涉及)
            2. llm-eval验证
            3. 端到端性能回归测试

## Reference material

1. agent_sys的使用样例：
    1. agent_sys/examples/demo: 可以真实运行的3节点任务
    2. agent_sys/examples/demo2:  一个算法题目出题到评审的7+节点任务
    3. agent_sys/examples/single_real_task: 一个使用infera + sglang在gfx942机器上运行qwen模型的任务。
2. glm5.3-flash使用infera跑通实例：
    1. smci355-ccs-aus-n02-29.prov.aus.ccs.cpe.ice.amd.com或 smc300x-ccs-aus-a16-10.prov.aus.ccs.cpe.ice.amd.com:/apps/yihou/packups/glm53flash.mix.packup_20260830
3. Optimus-AgenticBench使用样例：
    1. 如agent/workloads/glm52_crxx_caseA.fix.yaml， Optimus-AgenticBench可以设定cache hit rate, 模拟agent负载对指定大模型服务进行压测。
    2. smci355-ccs-aus-n02-29.prov.aus.ccs.cpe.ice.amd.com或 smc300x-ccs-aus-a16-10.prov.aus.ccs.cpe.ice.amd.com
    3. /apps/yihou/packups/mix.latency.packup_20260806.spur
    4. /apps/yihou/packups/mix.stress.packup_20260806.spur

## ISSUES

agent_sys是一个初步搭建的multi-agent工作系统，因此对本次任务的目的和注意事项进行额外说明：

1. 必须关闭权限校验模块
2. 指定agent back end为claude code sdk，自行提供api key和endpoint:
    1. "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/Anthropic"
    2. "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: ${从llm.amd.com生成获取}
3. agene_sys刚开发完成，可能有诸多bug, 如果遇到：
    1. 请分析确认为真实bug，然后记录
    2. 尽量尝试绕过bug完成本期目标
    3. 等待确认后修复（有足够信心为bug且在不违背设计的情况下可自行修复）。
4. 过程中不要过于纠结个别validator的校验方式和结果，而是focus到把整个任务跑通有稳定输出的问题上。
5. infera llm e2e性能优化kick off report: xxx

## RULES for developer[not agent sys or task package]

1. 开发使用dev.yihou.aiopt.task_package分支，pr142, 活动范围限制在agent_sys/examples/llm_e2e_performance_optimization/
    1. 如发现bug需要修复，请先在agent_sys/examples/llm_e2e_performance_optimization/temp/bugs/下记录，以便其他查看。
2. 人类需使用agent team进行工作（不是指agent_sys体系，使用cursor请无视）： claude update  &&   export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1  
3. 实验成果请使用agent_sys/examples/llm_e2e_performance_optimization/temp/claude_code_skill_used_by_human/experiment-result-packup技能packup
4. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
5. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
6. Use English when work, Chinese only for report to user.
