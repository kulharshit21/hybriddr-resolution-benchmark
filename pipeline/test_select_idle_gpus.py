import unittest

from select_idle_gpus import parse_csv, select_idle


class IdleGpuTests(unittest.TestCase):
    def stats(self):
        return [
            {'index': str(i), 'uuid': f'u{i}', 'memory_used_mib': '5',
             'memory_free_mib': '40400', 'utilization_percent': '0'}
            for i in range(8)
        ]

    def test_selects_preferred_then_sorted_idle_devices(self):
        self.assertEqual(select_idle(self.stats(), {'u3'}, 5, [7, 6, 5, 4, 3]),
                         [7, 6, 5, 4, 0])

    def test_rejects_memory_utilization_and_process_occupancy(self):
        rows = self.stats()
        rows[0]['memory_used_mib'] = '101'
        rows[1]['memory_free_mib'] = '37999'
        rows[2]['utilization_percent'] = '6'
        self.assertEqual(select_idle(rows, {'u3'}, 5), [])

    def test_csv_parser_trims_fields(self):
        self.assertEqual(parse_csv('0, GPU-a, 5\n'), [['0', 'GPU-a', '5']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
