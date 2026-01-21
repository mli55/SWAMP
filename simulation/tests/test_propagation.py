import sys
import pathlib
import unittest
import numpy as np

# Ensure the repo root is on sys.path for in-place imports
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rfspsim.media.medium import Medium
from rfspsim.media.interface import fresnel_rt, reflection_coeff
from rfspsim.propagation.planar import refracted_two_segment
from rfspsim.propagation.channel_builders import (
    build_surface_scatter_taps,
    build_target_reflection_taps,
)


class PropagationBasicsTest(unittest.TestCase):
    def test_fresnel_normal_incidence_matches_impedance_ratio(self):
        air = Medium("air", epsilon_r=1.0, mu_r=1.0)
        soil = Medium("soil", epsilon_r=4.0, mu_r=1.0)

        theta = 0.0
        g_te, t_te = fresnel_rt(air, soil, theta, pol="TE")
        g_tm, t_tm = fresnel_rt(air, soil, theta, pol="TM")

        eta1, eta2 = air.eta, soil.eta
        expected_gamma = (eta2 - eta1) / (eta2 + eta1)
        expected_tau = (2.0 * eta2) / (eta2 + eta1)

        np.testing.assert_allclose(g_te, expected_gamma, atol=1e-12)
        np.testing.assert_allclose(g_tm, expected_gamma, atol=1e-12)
        np.testing.assert_allclose(t_te, expected_tau, atol=1e-12)
        np.testing.assert_allclose(t_tm, expected_tau, atol=1e-12)

    def test_refracted_two_segment_vertical_alignment(self):
        # Crossing the interface vertically should hit (0,0) with zero angles
        p_air = np.array([0.0, -0.1])
        p_soil = np.array([0.0, 0.2])
        v = 3e8

        cross, L1, L2, theta_i, theta_t = refracted_two_segment(
            p_air, p_soil, v, v, interface_z=0.0
        )

        np.testing.assert_allclose(cross, [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(L1, 0.1, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(L2, 0.2, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(theta_i, 0.0, places=12)
        self.assertAlmostEqual(theta_t, 0.0, places=12)

    def test_target_reflection_delay_and_gain_without_fresnel(self):
        # Disable Fresnel to directly check geometric delay and 1/L^2 decay
        air = Medium("air", epsilon_r=1.0, mu_r=1.0)
        soil = Medium("soil", epsilon_r=1.0, mu_r=1.0)

        tx = np.array([-0.1, -0.05])
        rx = np.array([0.1, -0.05])
        target = np.array([[0.0, 0.2]])
        per_point_area = 0.01

        res = build_target_reflection_taps(
            tx,
            rx,
            target,
            air=air,
            soil=soil,
            per_point_area=per_point_area,
            target_reflectivity=1.0 + 0j,
            include_fresnel=False,
        )

        delays = res["delays"]
        gains = res["gains"]

        _, L_air_in, L_soil_in, _, _ = refracted_two_segment(
            tx, target[0], air.v, soil.v, interface_z=0.0
        )
        _, L_soil_out, L_air_out, _, _ = refracted_two_segment(
            target[0], rx, soil.v, air.v, interface_z=0.0
        )

        L_total = L_air_in + L_soil_in + L_soil_out + L_air_out
        expected_delay = L_total / air.v  # air.v == soil.v in this setup
        expected_gain = per_point_area / (L_total**2 + 1e-12)

        np.testing.assert_allclose(delays[0], expected_delay, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gains[0], expected_gain, rtol=1e-12, atol=1e-12)

    def test_surface_gain_scales_with_fresnel_reflection(self):
        air = Medium("air", epsilon_r=1.0, mu_r=1.0)
        soil = Medium("soil", epsilon_r=4.0, mu_r=1.0)

        tx = np.array([0.0, -0.2])
        rx = np.array([0.0, -0.2])
        surf_pt = np.array([[0.0, 0.0]])

        base = build_surface_scatter_taps(
            tx,
            rx,
            surf_pt,
            air=air,
            soil=soil,
            per_point_length=1.0,
            surface_reflectivity=1.0,
            include_fresnel=False,
        )
        fresnel_scaled = build_surface_scatter_taps(
            tx,
            rx,
            surf_pt,
            air=air,
            soil=soil,
            per_point_length=1.0,
            surface_reflectivity=1.0,
            include_fresnel=True,
            pol="avg",
        )

        theta = 0.0
        gamma = reflection_coeff(air, soil, theta, pol="avg")
        np.testing.assert_allclose(
            fresnel_scaled["gains"][0],
            base["gains"][0] * gamma,
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
