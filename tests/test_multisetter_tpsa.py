"""MultiSetter on elements whose fields hold TPSA series."""
import madng_tpsa
import numpy as np
import xtrack as xt


def test_multisetter_built_before_promotion_reads_the_series_value():
    line = xt.Line(elements={'quad': xt.Quadrupole(length=1., k1=.4)})
    line.build_tracker()
    setter = xt.MultiSetter(line, ['quad'], 'k1')
    line['quad'].k1 = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1).param(1, .4)
    np.testing.assert_array_equal(setter.get_values(), [.4])


def test_multisetter_can_be_built_after_promotion():
    line = xt.Line(elements={'quad': xt.Quadrupole(length=1., k1=.4)})
    line.build_tracker()
    line['quad'].k1 = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1).param(1, .4)
    np.testing.assert_array_equal(xt.MultiSetter(line, ['quad'], 'k1').get_values(), [.4])


def test_multisetter_write_keeps_the_parameter_of_another_field():
    line = xt.Line(elements={'quad': xt.Quadrupole(length=1., k1=.4, k1s=.25)})
    line.build_tracker()
    setter = xt.MultiSetter(line, ['quad'], 'k1')
    line['quad'].k1s = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1).param(1, .25)

    setter.set_values(np.array([.7]))

    assert line['quad'].k1.const_part == .7
    np.testing.assert_array_equal(line['quad'].k1s.param_grad(), [1.])


def test_multisetter_writes_plain_values_again_after_tpsa_is_disabled():
    line = xt.Line(elements={'quad': xt.Quadrupole(length=1., k1=.4)})
    line.build_tracker()
    setter = xt.MultiSetter(line, ['quad'], 'k1')
    line['quad'].k1 = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1).param(1, .4)
    line.disable_tpsa_elements()

    setter.set_values(np.array([.8]))

    assert line['quad'].k1 == .8


def test_multisetter_on_a_slice_reaches_its_parent_series():
    line = xt.Line(elements={'quad': xt.Quadrupole(length=1., k1=.4, k1s=.25)})
    line.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(2, mode='thick'), element_type=xt.Quadrupole)])
    line.build_tracker()
    slices = [name for name in line.element_names if isinstance(line.get(name), xt.ThickSliceQuadrupole)]
    setter = xt.MultiSetter(line, slices, ('_parent', 'k1'))
    line['quad'].k1s = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1).param(1, .25)

    setter.set_values(np.array([.7, .7]))

    assert line['quad'].k1.const_part == .7
    np.testing.assert_array_equal(line['quad'].k1s.param_grad(), [1.])


def test_line_attr_reads_a_promoted_element():
    line = xt.Line(elements={'quad': xt.Quadrupole(length=1., k1=.4), 'drift': xt.Drift(length=.2)})
    line.build_tracker()
    line['quad'].k1 = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1).param(1, .4)
    np.testing.assert_allclose(line.attr['k1l'], [.4, 0.], rtol=1e-15)
