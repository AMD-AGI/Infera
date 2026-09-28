"""Guard GPU-subset checks against HIP/SMI numbering differences."""
import unittest
from gpu_inventory import match_cards


class MappingTests(unittest.TestCase):
    def test_pci_mapping_can_cross_half_node_groups(self):
        cards={f'card{i}':{'PCI Bus':f'0000:{i:02x}:00.0'} for i in range(8)}
        order=[7,0,2,1,3,4,6,5]
        hip={str(i):cards[f'card{card}']['PCI Bus'].upper() for i,card in enumerate(order)}
        mapping=match_cards(hip,cards)
        self.assertEqual([mapping[i] for i in range(4)],['card7','card0','card2','card1'])
        self.assertEqual(set(mapping.values()),set(cards))

    def test_missing_pci_does_not_mean_free_memory(self):
        with self.assertRaises(KeyError):match_cards({'0':'0000:01:00.0'},{'card0':{'PCI Bus':'0000:02:00.0'}})

    def test_duplicate_pci_is_rejected(self):
        with self.assertRaises(AssertionError):match_cards({'0':'0000:01:00.0','1':'0000:01:00.0'},{'card0':{'PCI Bus':'0000:01:00.0'}})
        with self.assertRaises(AssertionError):match_cards({'0':'0000:01:00.0'},{'card0':{'PCI Bus':'0000:01:00.0'},'card1':{'PCI Bus':'0000:01:00.0'}})


if __name__=='__main__':unittest.main()
