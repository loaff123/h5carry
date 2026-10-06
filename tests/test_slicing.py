import importlib.util
import unittest

class SlicingTests(unittest.TestCase):
    def function(self):
        self.assertIsNotNone(importlib.util.find_spec('h5carry.slicing'),'bounded rectangular tiling missing')
        from h5carry.slicing import iter_transfers
        return iter_transfers
    def test_offset_bounded_c_order_and_chunk_boundary(self):
        blocks=list(self.function()([1,2],[3,11],4,32,chunks=(2,4)))
        covered=[]
        for left,right in blocks:
            self.assertLessEqual(__import__('math').prod(s.stop-s.start for s in left)*4,32)
            self.assertEqual(left[-1].start//4,(left[-1].stop-1)//4)
            covered += [(i,j) for i in range(left[0].start,left[0].stop) for j in range(left[1].start,left[1].stop)]
            self.assertEqual([s.start for s in right],[left[0].start-1,left[1].start-2])
        self.assertEqual(covered,[(i,j) for i in range(1,3) for j in range(2,11)])
    def test_huge_prefix_is_lazy_and_empty_has_no_blocks(self):
        iterator=self.function()([0]*32,[2**20]+[1]*30+[2],1,1024)
        self.assertEqual(next(iterator)[0][0],slice(0,1));self.assertEqual(next(iterator)[0][0],slice(1,2))
        self.assertEqual(list(self.function()([5,3],[5,8],8,1024)),[])
    def test_oversized_native_chunk_refuses_before_any_tile(self):
        from h5carry.model import CarryError
        with self.assertRaises(CarryError):next(self.function()([0,0],[1,1],8,64,chunks=(4,4)))
