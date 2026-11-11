import math
import unittest
import tempfile
from pathlib import Path
from wam_astra.navigation_memory import displacement, select_candidates, rank_windows, preceding_rows


class NavigationMemoryTests(unittest.TestCase):
    def test_causal_state_selection_counts_clock_error_and_unsorted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'state.csv'
            path.write_text('relative_time_s,clock_estimated_error_ns,source_row\n1,0,a\n2,100000000,b\n1.9,0,c\n',encoding='utf-8')
            rows=preceding_rows(path,[.5,1,2,2.1,3])
            self.assertEqual([r['source_row'] if r else None for r in rows],[None,'a','c','b','b'])
    def test_body_frame_uses_start_yaw_and_measured_values(self):
        start=dict(position_x=1,position_y=2,position_z=3,yaw=math.pi/2)
        end=dict(position_x=1,position_y=2.3,position_z=3,yaw=math.pi/2)
        d=displacement(start,end,'go2')
        self.assertAlmostEqual(d['body_xyz_m'][0],.3)
        self.assertAlmostEqual(d['body_xyz_m'][1],0)
        self.assertEqual(d['semantics'],'measured_state_change_not_command')

    def test_missing_or_nonfinite_state_does_not_create_motion(self):
        self.assertIsNone(displacement(None,{},'uav'))
        self.assertIsNone(displacement({'position_x':float('nan')},{},'go2'))

    def test_selection_preserves_diversity_and_deduplicates(self):
        rows=[dict(start_s=i*3,end_s=(i+1)*3,motion={'go2':{'distance_m':i,'yaw_delta_rad':0},'uav':None}) for i in range(10)]
        selected=select_candidates(rows)
        self.assertIn(rows[-1],selected)
        self.assertEqual(len(selected),len({r['start_s'] for r in selected}))
        self.assertIn(rows[0],selected)

    def test_retrieval_excludes_entire_current_episode_and_unknown_outcomes(self):
        rows=[dict(episode_id=e,window_id=e+str(i),motion={'go2':{'distance_m':i}},task_success='unknown') for e,i in [('current',100),('other',1)]]
        hits=rank_windows(rows,robot='go2',exclude_episode='current',k=2)
        self.assertEqual([h['episode_id'] for h in hits],['other'])
        self.assertEqual(hits[0]['task_success'],'unknown')

if __name__=='__main__':unittest.main()
