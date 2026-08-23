import unittest

import numpy as np

from utils.contact_diagnostics import (
    deform_surface_samples,
    frame_contact_metrics,
    infer_contact_windows,
    label_vertex_components,
    sample_surface_template,
    select_low_vertices,
)


class ContactDiagnosticsTest(unittest.TestCase):
    def test_select_low_vertices(self):
        vertices = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=float)
        np.testing.assert_array_equal(select_low_vertices(vertices, 0.5), [0, 1])

    def test_surface_samples_follow_deformation(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2]])
        samples = sample_surface_template(vertices, faces, sample_count=20, seed=3)
        points_before, normals_before = deform_surface_samples(vertices, faces, samples)
        translated = vertices + np.array([0.0, 2.0, 0.0])
        points_after, normals_after = deform_surface_samples(translated, faces, samples)
        expected_translation = np.tile([0.0, 2.0, 0.0], (len(points_before), 1))
        np.testing.assert_allclose(points_after - points_before, expected_translation)
        np.testing.assert_allclose(normals_after, normals_before)

    def test_label_vertex_components(self):
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        count, labels = label_vertex_components(faces, np.arange(6), vertex_count=6)
        self.assertEqual(count, 2)
        self.assertEqual(sorted(np.bincount(labels).tolist()), [3, 3])

    def test_frame_metrics_detect_near_candidates(self):
        surface = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        normals = np.array([[0, 1, 0], [0, 1, 0]], dtype=float)
        actor = np.array([[0, 0.01, 0], [1, 0.20, 0], [4, 4, 4]], dtype=float)
        metrics = frame_contact_metrics(actor, np.array([0, 1]), surface, normals, 0.05)
        self.assertEqual(metrics["near_candidate_count"], 1)
        self.assertAlmostEqual(metrics["candidate_min_distance"], 0.01)
        self.assertAlmostEqual(metrics["mean_near_oriented_gap_proxy"], 0.01)

    def test_infer_and_merge_padded_windows(self):
        counts = np.array([0, 2, 2, 0, 0, 1, 0])
        self.assertEqual(infer_contact_windows(counts, padding=1), [(0, 6)])
        self.assertEqual(infer_contact_windows(counts, padding=0, min_count=2), [(1, 2)])
        self.assertEqual(infer_contact_windows(np.zeros(4), padding=1), [])


if __name__ == "__main__":
    unittest.main()
