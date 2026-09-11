"""Regression for the missing first rejection proposal in fake PD handoff."""
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

# Load the exact module file; use real Torch/EagleDraftInput, only fake the
# scheduler and runtime settings which normally need a running process group.
path = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location('handoff_under_test', path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProposalTest(unittest.TestCase):
    def run_handoff(self, fake, rejection, overlap):
        settings = SimpleNamespace(speculative_eagle_topk=1,
                                   speculative_num_steps=5,
                                   enable_multi_layer_eagle=False,
                                   speculative_use_rejection_sampling=rejection)
        req = SimpleNamespace(output_topk_p=torch.tensor([0.0]),
                              output_topk_index=torch.tensor([7]),
                              hidden_states_tensor=torch.zeros(8),
                              output_dsa_topk_indices=None)
        batch = SimpleNamespace(reqs=[req], device='cpu',
                                model_config=SimpleNamespace(vocab_size=16),
                                enable_overlap=overlap,
                                req_pool_indices=torch.tensor([0]),
                                seq_lens=torch.tensor([32]))
        received = []
        relay = SimpleNamespace(publish=lambda *a: None,
                                stash=lambda indices,payload: received.append(payload))
        with patch.object(module, 'get_spec', return_value=settings):
            with patch.object(module, 'get_disagg', create=True,
                              return_value=SimpleNamespace(disaggregation_transfer_backend='fake' if fake else 'mooncake')):
                out = module.build_eagle_disagg_draft_input(batch, torch.tensor([0]), relay)
        return out, received

    def test_fake_rejection_has_normalized_first_proposal(self):
        out, _ = self.run_handoff(True, True, False)
        self.assertIsNotNone(out.draft_probs)
        self.assertEqual(tuple(out.draft_probs.shape), (1,16))
        self.assertEqual(out.draft_probs[0,7].item(), 1.0)
        self.assertEqual(out.draft_probs.sum().item(), 1.0)
        self.assertEqual(out.topk_p.item(), 1.0)
        torch.stack([out.draft_probs]*5, dim=1)

    def test_fake_proposal_is_carried_into_overlap_relay(self):
        out, received = self.run_handoff(True, True, True)
        self.assertIsNotNone(out.draft_probs)
        self.assertIs(received[0].draft_probs, out.draft_probs)

    def test_real_transfer_remains_unchanged(self):
        out, _ = self.run_handoff(False, True, False)
        self.assertIsNone(out.draft_probs)
        self.assertEqual(out.topk_p.item(), 0.0)

    def test_no_rejection_keeps_old_path(self):
        out, _ = self.run_handoff(True, False, False)
        self.assertIsNone(out.draft_probs)


unittest.main()
