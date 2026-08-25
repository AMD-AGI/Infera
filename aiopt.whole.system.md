## category

开发

## Brief

本项目旨在设计一个针对指定任务流录制执行的multi-agent system。核心为借鉴编程中的函数思路。将任务(task)封装为标准的输入输出(handoff), 主体由agent执行。从而保证工作交付的质量。

请结合用户思考产出，research, 完善spec, research，design, 实现。

具体用户思考产出如下：

1. system = task graph = graph of <handoffs, agent>
2. task
    1. task = <handoffs, agent>，表示：
        1. 什么任务
        2. 输入输出是什么
        3. 配套的agent spec是什么。
    2. task有scheduler(已经实现)
    3. task可以有子图嵌套，每个task有一个start entry subtask和/end entry subtask（可以是自身）。当start被调用，说明子图开始，end结束，标明子图结束。
    4. 整个系统的任务有一个system whole task. 他展开是一个子图。
    5. 目前只支持内部graph statically defined的task，主要因为本系统第一阶段主要用于流程固定的工作。
    6. task scheduler、agent、task、task runner的关系：
        1. task是一个数据记录类，维护一个task的状态。
        2. task scheduler不感知agent
        3. task runner管理task状态和内部语义一致性，agent、task scheduler与task runner相互交互，更新task状态（可能需要hook）
    7. task展开后主要分为3部分
        1. input validations
            1. 不重复校验单个handoff
            2. 一般为空
            3. 多个handoff综合的校验逻辑写在这里
        2. main
        3. output validations
            1. 写在这里往往是因为validation需要的环境往往和主task一致。
        4. 每个handoff都至少配备一个validator, 除非在命令行特殊开关的情况下，允许没有。
        5. subtask graph在main中展开
3. handoff:
    1. handoff是一个模块的输入/输出
    2. 一些已思考到的必要字段schema
        1. 日期
        2. runtime的产出应该有md5/hash/任选一种校验机制，防串改。
        3. 代码库中已有条目: version等
        4. content
            1. executable:
                1. null/command/recipe & scripts
            2. result:
                1. result schema
                2. result content
        5. validator list
    3. 参考task可知，handoff的spec虽然独立成folder，但是<handoff spec set, task spec, agent spec>这四者个往往成一个组合（实际上一般来说task spec和handoff spec是强绑定关系。），我们权且成为一个closure， closure需提前定义好。因为系统暂不支持dynamic task spec。所以handoff的spec也是提前定义好的。(也就是系统本质上只支持录制回放。但实际上task spec可以设置为), closure本身只是一个抽象的表格/wrapper系统，不提供额外的语义。
    4. handoff spec需要一份yaml文件，有schema约束，记录handoff必要的信息
    5. handoff必须”可校验”，也就是说每一个handoff配套若干个validator, 每个handoff至少有一个validator.
        1. 由于validators是检验handoff的唯一标准，当一个handoff的种类被提交入库时，除了给出yaml和validator以外，还应重点review其validator能多大程度上有效的校验handoff
        2. handoff必须标准化，包括包含哪些东西，每个东西都在哪里，因此validator也会标准化填空。固化掉流程部分，补全内容。内容往往有Jsonnet或者其他这种配置即文档的东西组装，不允许现场修改接入代码。可通过config注入template或缺的，待补全的指定组件代码。
        3. 每个handoff spec必须写明和自身相关的validator有哪些（包括只需要自己的，和需要自己和别人的）
    6. handoff按照作用域和声明周期的tag含义有以下分离：
        1. fixed handoff （一个完整录制的task规定的输入输出）
            1. required
            2. optional
        2. addons handoff (临时注入的handoff, 并不在规定输入以内，但是对完成任务有帮助，可以是用户注入的，也可以是其他agent)
            1. temp、local/run related
            2. knowledge handoff（长期积累的知识经验， 作为agent specialist的重要资料）
4. validator
    1. validator也是特殊的task的spec，首先必须是单节点的，内部不能包含subtask，在实际运行中会成为一个task，它接受一个/多个handoff，输出一个dict[key: handoff, value:bool/score]的字典, 并且自身作为task的input validation应为空。
    2. validator内部往往由agent任务指示、代码、config/info yaml文件三类组成，总体划分为三要素：<input, process, result>，result分两种：bool或score, 目前只支持bool. 因此一个具体的validator应为一份固定的流程加一堆待补全的空缺(已由yaml文件制定好空缺位置)
        1. input: 从handoff来，每一个将具体内容填入具体key, 如何从handoff中拿去并放入对应的位置。
        2. process block: 模板留白的一些流程，输入和输出格式已事先定义。
            1. 叶节点的validator不允许由自己的留白block
        3. 根据上述空缺：validator可以有自己的template，并且无限递归（但实际上目前会限制依赖层数， 比如3层）：
            1. 叶节点的validator总是绑定到一个或者多个handoff种类上的，这是一个二者都维护记录的映射。
    3. validator系统不对某一类或者可被归为一类的测试代码的抽象、框架化负责：
        1. 这些代码应该被框架化，但不由validator系统负责。由对应归属的外部test系统负责。
    4. validator的可信度问题：
        1. validator根据其检验逻辑，检验标准。通常需要：
            1. 校验逻辑，（必须外界制定）：
                1. 外界编写，程序化：高可信（外界又可以分为静态、流程其他task产生的动态）
                2. 外界编写，agent:  中等可信
            2. 一个校验标准（必须外界制定）：
                1. 可明确量化：高可信 。
                2. 难以量化：低可信。
        2. 因此validator可被分为两类：
            1. strong: 强标准、 可快速短期检验的。
                1. 有量化检验方法。
                2. 有result和gt或二者的产出办法。
                3. 风险可最终闭环的。
                4. long term strong:
                    1. 校验链路/反馈周期长: 整个系统执行完后，结合其他handoff.
            2. weak: 没有清晰判据/短期判据（评估周期长、风险敞口），但是端到端结果是可以评估的
                1. 没有量化标准
                2. 风险敞口
5. 在具体的大模型e2e优化系统里，每一个task的handoff和validator举例分析
    1. 人类给出的客户<deploy config, workload, SLA>:
        1. weak：
            1. 是否sota, 是否生产级别。agent无法给出准确大概。只能按照以下资料分析
                1. 网络公开知识经验。
                2. 自身/用户之前知识。
                3. 个别可由开源公开/客户担保。
    2. e2e 运行方法hand off
        1. strong:
            1.  正确性：curl/ping/简单压测。标准可量化，过程可程序化，由外部指定。
        2. weak:
            1. 是否准确打开所有性能最优knob:
                1. 由agent结合各种knowledge(sglang/vllm代码/网络/本地积累)给出分析。
                2. 但个别可由是否达到预期性能达到strong级别校验。
    3. trace getter
        1. strong:
            1. 可用性：chema校验、可读取性、关键字段是否存在
            2. 流程运行完后的feedback, 
            3. 一些自洽性的检验
                1. 可以有一些大致百分比的加和校验
                2. mock扩大/缩小/删除/添加耗时， 验证变化情况。
                    1. 源码级别重新测试
                    2. trace级别当场校验。
        2. weak:
            1. 完整性：
                1. 结合模型结构、代码分析，给出完整性结论，不可程序化，个别模型可强行适配
            2. 结果耗时是否符合预期：
                1. profiler的额外负载是多少？只能由经验给出。
    4. trace analysis、extractor
        1. strong:
            1. top-k：不做headroom分析的话基本可信。
            2. 抠出来的代码：
                1. 正确性测试代码，可以可信（可以截取输入输出）。
                2. 时间测试代码：可信（测试方法、，结果标准由别人给出）
                3. 流程运行完后的feedback
        2. weak:
            1. 如果做headroom分析和更多roofline bottleneck分析
            2. 结合pipeline/overlap的trace, 首轮分析
    5. optimized kernel handoff
        1. strong:
            1. 对拍代码和正确性校验器，应该由隐藏输入（由前task提供）
            2. 性能评估器（由前task提供）
            3. 1和2都可以代码模板。
            4. 简单来说运行方法、检验方法、校验内容完全由别人提供。
        2. weak:
            1. 优化质量
    6. 接入完成handoff
        1. strong:
            1. eval
            2. bench
            3. 运行方法，检验方法，检验内容，检验标准，都由别人定义。
    7. global validator
        1. 回头看trace/analysis是否合理了
    8. validator本身的folder应该本良好的管理，以便不断更新:
        1. 叶节点validator
            1. 在文件夹名上.leaf.标识
            2. 内部应有其handoffs的相对路径软链
        2. templates validator
            1. 在文件夹名上.template.
        3. validator应该有一个tag dict, 每个tag有key: value
        4. validator本身应该有个code folder/registory/mgr。记录全局有哪些validator, 每个validator应该由多个tag: 包括不同维度的类别。和在当前工作系统中会被谁用到，被谁用到过。
        5. 能想到goal validator
            1. weak: 
                1. 由一个独立的agent，拿到你的输出和任务定义，评价你的工作是否完成。
                2. cheat validator, 检查你是否作弊? 
        6. validator遵循的原则。
            1. 校验者与生产者context分离
                1. 通过hook锁死校验者/生产者agent context分离。
            2. 能代码化的流程尽量代码化，去掉agent或者让agent起到一个实习生填空组装流程的作用。
6. ai agent是目前支持的agent， agent也可以由人/程序扮演，只要handoff是标准化的.
    1. agent可以为task展开子图。
    2. agent对象需要一份yaml文件，有schema约束，记录handoff必要的信息，包含但不限于：
        1. permission list:
            1. 可以读取的handoff
            2. 可以访问的workspace/playground
        2. env
7. env_mgr的职责应该被拓宽，管理所以与linux系统的交互，包括（因此应该被重新涉及，已有模块可以复用，并保证涉及足够解耦）：
    1. 所有storage和持久化
    2. agent自身的工作环境（local）
    3. agent用来做实验的开发环境的基础使用方法和规范。(remote) （虽然local和remote可以是一个）
    4. 具体的，env mgr需要划分子模块以管理以下内容
        1. 有自己的metadata和配置，比如上述，会被持久化（可自动检测，可由指定task生成）比如下列各项的一些映射关系，基本配置等。
        2. file system manager模块：
            1. 域注册机制（reload/create）如：
                1. handoff storage
                2. playground
                3. workspace
            2. 域内权限管理机制
                1. 通过prefix, substr, suffix等检查的
            3. local和remote的映射方法和storage同步机制
                1. 强：相同nfs映射
                2. 强：相同mount映射
                3. 弱：rsync （需显式调用）
        3. remote的登录方法：
            1. ssh/docker exec
        4. agent local:
            1. workspace:
                1. 一个工作主仓库开辟出来的worktree
                2. 内部自行clone其他依赖仓库
                3. 一般来说工作不允许
            2. playground一个临时工作区, 存放中间文件，入运行时log，调试脚本、过程进展记录等。一般来说一个agent的history(由agent harness管理，如claude code) + playground + workspace 就可以resume一个agent。比如nfs上的用户指定的root文件夹下的一个子文件夹如果没有声明nfs，则应是workspace下的一个子目录
            3. handoff storage
        5. agent remote（如果存在）: 同agent local。
            1. 对应远端workspace
            2. 对应远端playground
            3. 对应远端handoff storage
        6. 原则上：
            1. agent不论在在本地还是远端，都不允许写出外露。
            2. 读取稍微松一点。被管理的文件系统（hook禁止的），其余应是允许的。
        7. env_mgr不负责感知和维护程序(task)运行时环境（只对agent所需的依赖负责），比如sglang/vllm/infera所需的环境和前后的一致性。这些：
            1. 一致性和规定由handoff和validator的具体实现负责。
            2. 额外的部署指南和集群使用规范作为knowledge handoff注入, 并且env_mgr在发现环境和自身deploy时也会使用。
                1. 所以也就是会有一些system level的task/agent/handoff.
        8. agent与具体实现解耦，比如agent底层(backend)可以是claude code, cursor, codex。
            1. backend自行组织的multi-agent系统不由系统感知。系统始终认为交给backend的工作是由一个agent完成的。也就是说系统的agent只负责“大的”agent节点。不管理细节。
            2. backend也需要简单的封装，以暴露统一的接口，比如get history, set rule, add hook等。但最初可以保证解耦的情况下用非常naive的实现。
            3. 结合2：agent自身需要提供observe和interactive的接口：
                1. 抓取历史
                2. 打断
                3. 消息队列追加
            4. agent应该有一个log tool call，logger有不同级别，agent根据情况输出[debug/info/warning/error]. log的发起原因可以来自于系统级别的log要求、自身log规范的要求，相关skill/rule/hook的log要求。
        9. handoff storage entry: 一个读取和存放handoff的接口。
            1. handoff应该是本地依赖无关的（不应包含本地信息，声明硬软件依赖，docker image等，可以任选合理机器运行）
            2. agent在使用handoff前，应由handoff storage读取 copy进playground, 从而使用handoff副本
            3. agent只能访问自身被允许的entry
        10. 另：env_mgr职责：
            1. 在agent工作前为agent准备好工作环境的一切，包括但不限于
                1. handoff
                2. workspace、playground
                3. 所有storage都必须由对应的permission hook
                    1. 比如文件系统，应该在扁平化组织目录的同时，由共同前缀，比如${some_root}/agent-handoff-permission-zone-prefix.xxxx。从而使得hook可以快速通过agent-handoff-prefix判断是否为带权鉴领域，然后根据xxx对比config中的字段，校验访问权限
                    2. 包括远端工作规范，如登录到container内部或者pod内部。
8. others暂时忽略。
9. 汇总一下这里的对象和其registery都有, 其中每个对象都有一个static的spec，一个运行时的独一无二的uuid标识的对象，一个runtime的mgr和一个spec的registery，以及一个单独的文件夹，用于存放每类的预定义的spec.
    1. handoff
    2. validator
    3. task
    4. agent

## Goal

1. 通过brief、materials理解需求和初步设计, 调研、分析是否:
    1. 有不完善的地方。
    2. 有需要改进的地方。
    3. 需要更多的设计和模块
2. research、design.
3. coding & test.

最终完成上述系统

1. 接入claude agent sdk.
2. demo graph通过命令行跑通。足够简单足够快，但是要体现task graph和sub graph.

## Task breakdown

1. 是否需要plan或者loop

## reference material

1. 已经部分完成的env_mgr和task scheduler代码：
2. 首个需要落地的首个场景相关项目文档：

## rules

1. 所有内容写入agent_sys子文件夹，并合适组织代码。
2. 要求依次交付spec, design doc、test & code. 每个环节都暂停由用户review.
3. 要求设计和实现尽量简洁。有成熟的解决方案使用成熟的解决方案：

```markdown
1. First research whether a mature, widely adopted standard interface / library / CLI tool
   already exists.
2. If a best practice or de facto standard implementation exists, **use it directly** —
   do not rewrite it.
3. If a simple partial use or a wrapper of a standard interface / library / CLI tool can implement it, use the **simple wrapper strategy**
3. Record in `modules/<name>/README.md`: which library/tool was chosen and why.
4. Only implement it yourself when no existing solution fits the requirement, and explain
   the rationale.
```

1. 严格遵循用户级别CLAUDE.md的工作规范，如有调整，请交互式询问用户。
2. 先research、gather information、analysis. 再plan， 再创建子workspace(保证所有的临时实验活动都在其中进行)，再创建CLAUDE.md(备份原先的)，再开始工作。
3. Use English when work, Chinese only for report to user.
