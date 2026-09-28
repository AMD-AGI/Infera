import json,subprocess,datetime
ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts']
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
for node,name in [('smci355-ccs-aus-n10-29','prefill'),('smci355-ccs-aus-n03-33','decode')]:
 cmd=f'docker inspect --format "{{{{.State.Status}}}} {{{{.State.ExitCode}}}} {{{{.State.StartedAt}}}}" llying-session-31999-{name}-0; docker logs --tail 6 llying-session-31999-{name}-0'
 p=subprocess.run(ssh+[node,cmd],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
 lines=p.stdout.replace('\r','\n').splitlines();print(name,'\n'.join(lines[:1]+lines[-5:]))
