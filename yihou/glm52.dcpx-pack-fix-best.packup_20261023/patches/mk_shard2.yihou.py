# mk_shard2 (task dcpx7b 2026-10-22): write conn.yihou-dcpshard2.py = overlay 93034eba91 mooncake/conn.py with one change in
# MooncakeKVManager.add_transfer_request: when YIHOU_DCP_SHARD=room and a destination of the room needs DCP relayout,
# pick the transfer queue by bootstrap_room instead of sum(dst session ports) (constant for a DP1 DCP decode -> every
# chunk of a prefill rank went to queue 0, py-spy dcpx6 A2). All chunks of one room still share a queue (ordering kept).
# Usage: python3 -I mk_shard.yihou.py <src conn.py> <dst file>
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = """        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)
"""
new = """        dst_infos = self.transfer_infos[bootstrap_room].keys()
        if _YIHOU_DCP_SHARD == "room" and any(
            getattr(self.decode_kv_args_table.get(session), "requires_dcp_relayout", False)
            for session in dst_infos
        ):
            # yihou dcpx7b: a DP1 DCP decode gives every room the same destination set, so the port sum is constant
            # and all chunks of this prefill rank would serialize on one worker. Shard by room instead; all chunks of
            # one room still land on the same queue. Hash the room: room % dp_size is the prefill DP rank
            # (data_parallel_controller.py:883), so plain room % 4 is again constant per prefill rank (dcpx7).
            shard_idx = ((bootstrap_room * 0x9E3779B97F4A7C15) >> 32) % len(self.transfer_queues)
        else:
            session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
            shard_idx = session_port_sum % len(self.transfer_queues)
"""
assert s.count(old) == 1, "anchor not found exactly once"
s = s.replace(old, new)
anchor = "\nlogger = logging.getLogger(__name__)\n"
assert s.count(anchor) == 1, "logger anchor"
s = s.replace(anchor, anchor + '_YIHOU_DCP_SHARD = os.environ.get("YIHOU_DCP_SHARD", "")\n')
assert "\nimport os\n" in s
open(dst, "w").write(s)
print("ok", dst)
