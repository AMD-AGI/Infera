# main

A non-leaf whose subgraph fans out to three producers and joins at merge.
emit_b deliberately writes rows missing a required field, so check_thing
fails it and merge never starts.
