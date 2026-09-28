import subprocess,unittest
from unittest.mock import Mock,patch
from allocation_ready import wait_for_ssh


class AdmissionTests(unittest.TestCase):
    @patch('allocation_ready.time.sleep')
    @patch('allocation_ready.subprocess.check_output',return_value='JobState=RUNNING PreemptTime=None')
    def test_transient_ssh_failure_is_retried(self,query,sleep):
        remote=Mock(side_effect=[subprocess.CalledProcessError(255,['ssh']),''])
        wait_for_ssh(remote,'node',1)
        self.assertEqual(remote.call_count,2)
        sleep.assert_called_once_with(5)

    @patch('allocation_ready.subprocess.check_output',return_value='JobState=CANCELLED PreemptTime=None')
    def test_lost_job_is_not_retried(self,query):
        with self.assertRaisesRegex(RuntimeError,'CANCELLED'):
            wait_for_ssh(Mock(side_effect=subprocess.CalledProcessError(255,['ssh'])),'node',1)

    @patch('allocation_ready.subprocess.check_output',return_value='JobState=RUNNING PreemptTime=2026-09-28T18:13:42')
    def test_preemption_grace_is_not_usable_admission(self,query):
        with self.assertRaisesRegex(RuntimeError,'preempted'):
            wait_for_ssh(Mock(side_effect=subprocess.CalledProcessError(255,['ssh'])),'node',1)


if __name__=='__main__':unittest.main()
