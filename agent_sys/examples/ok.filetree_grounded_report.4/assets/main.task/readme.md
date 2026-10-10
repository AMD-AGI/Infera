# main — the whole demo, as one task

This task runs nothing itself. Its work is its subgraph:

| | |
|---|---|
| `produce` | a program node. Walks this package and writes a `facts` manifest |
| `describe` | an agent node. Reads `facts` and writes a `summary` |
| `consume` | a program node. Would render the summary into a report |

It carries this readme and **no `entry.sh`**, and that is the point of it being
here: the rule is *`entry.sh` versus a subgraph*, not *a body versus a
subgraph*. A non-leaf still has to say what it is for, or it is a step nobody
can review.
