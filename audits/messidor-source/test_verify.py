import unittest
from verify import check_pairs
class Tests(unittest.TestCase):
    def test_complete_pairs_and_reject_duplicate(self):
        rows=[[f'a{i}.png',f'b{i}.jpg'] for i in range(874)]
        known={x.rsplit('.',1)[0] for row in rows for x in row}
        self.assertEqual(len(check_pairs(rows,known)),874)
        with self.assertRaises(ValueError):check_pairs(rows+rows[:1],known)
        with self.assertRaises(ValueError):check_pairs(rows[:-1],known)
    def test_filenames_do_not_infer_pairs(self):
        with self.assertRaises(ValueError):check_pairs([['a.png'],['b.png']],{'a','b'})
if __name__=='__main__':unittest.main(verbosity=2)
