"""Reproduce the native/public KV collision and check the reserved window."""
import importlib.util,os,unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('net',Path(__file__).parents[3]/'infera/common/net.py')
net=importlib.util.module_from_spec(spec);spec.loader.exec_module(net)

class Socket:
    def __init__(self,*args):pass
    def bind(self,address):pass
    def close(self):pass

class ReservationTest(unittest.TestCase):
    def test_observed_native_block_cannot_overlap_public_port(self):
        with patch.object(net.random,'randint',return_value=25539),patch.object(net.socket,'socket',side_effect=Socket):
            with patch.dict(os.environ,{'INFERA_NODEPORT_RANGE':'30000-32767'}):old=net.free_tcp_port_block(8)
            with patch.dict(os.environ,{'INFERA_NODEPORT_RANGE':'25000-32767'}):fixed=net.free_tcp_port_block(8)
        self.assertLessEqual(old,25541)
        self.assertLess(25541,old+8)
        self.assertLess(fixed+7,25000)

if __name__=='__main__':unittest.main()
