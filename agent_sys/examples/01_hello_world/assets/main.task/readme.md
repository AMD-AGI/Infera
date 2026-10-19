# main — hello world

This task runs nothing itself; its work is its subgraph.

| | |
|---|---|
| `env_facts` | a program task. A script measures this machine and publishes `env_facts` |
| `env_check` | an AI task. An agent re-checks each fact with its own command and publishes `env_report` |

A non-leaf carries a readme and no `entry.sh`: it still has to say what it is for.
