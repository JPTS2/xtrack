import xtrack as xt
import xobjects as xo

import numpy as np

env = xt.Environment()

components=[
    env.new('m1', 'Marker', at='q1@start'),
    env.new('q1', 'Quadrupole', length=2.0, anchor='start', at=1.),
    env.new('q2', 'q1', anchor='start', at=10., from_='q1@end'),
    env.new('s2', 'Sextupole', length=0.1, anchor='end', at=-1., from_='q2@start'),

    env.new('q3', 'Quadrupole', length=2.0, at=20.),
    env.new('q4', 'q3', anchor='start', at='q3@end'),
    env.new('q5', 'q3'),

    # Sandwitch of markers expected [m2.0, m2, m2.1.0, m2.1]
    env.new('m2', 'Marker', at='q2@start'),
    env.new('m2_0', 'Marker', at='m2@start'),
    env.new('m2_1', 'Marker', at='m2@end'),
    env.new('m2_1_0', 'Marker', at='m2_1@start'),
    env.new('m2_1_1', 'Marker'),

    env.new('m4', 'Marker', at='q4@start'),
    env.new('m3', 'Marker', at='q3@end'),
]
line = env.new_line(components=components)

line.get_table().show(cols=['name', 's_start', 's_end', 's_center'])

tt = line.get_table()

tt.show(cols=['name', 's_start', 's_end', 's_center'])

assert np.all(tt.name == np.array(
    ['drift_1', 'm1', 'q1', 'drift_2', 's2', 'drift_3', 'm2_0', 'm2',
       'm2_1_0', 'm2_1_1', 'm2_1', 'q2', 'drift_4', 'q3', 'm3', 'm4',
       'q4', 'q5', '_end_point']))
xo.assert_allclose(tt.s, np.array(
    [ 0. ,  1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. ,
       13. , 15. , 19. , 21. , 21. , 21. , 23. , 25. ]),
    rtol=0., atol=1e-14)
xo.assert_allclose(tt.s_start, np.array(
    [ 0. ,  1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. ,
       13. , 15. , 19. , 21. , 21. , 21. , 23. , 25. ]),
    rtol=0., atol=1e-14)
xo.assert_allclose(tt.s_end, np.array(
    [ 1. ,  1. ,  3. , 11.9, 12. , 13. , 13. , 13. , 13. , 13. , 13. ,
       15. , 19. , 21. , 21. , 21. , 23. , 25. , 25. ]),
    rtol=0., atol=1e-14)
xo.assert_allclose(tt.s_center, 0.5*(tt.s_start + tt.s_end),
    rtol=0., atol=1e-14)