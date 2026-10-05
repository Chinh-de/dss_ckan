import unittest
import numpy as np
from app.recommendation.dynamic_propagation import build_kg_dict, generate_user_triple_set
from app.recommendation.engine import recommendation_engine

class TestMultiDomainRecommendation(unittest.TestCase):
    def test_kg_dict_and_dynamic_propagation(self):
        sample_kg = np.array([
            [100, 0, 1001],
            [100, 1, 2001],
            [101, 0, 1001],
            [102, 1, 2001],
        ], dtype=np.int32)

        kg_dict = build_kg_dict(sample_kg)
        self.assertIn(100, kg_dict)
        self.assertEqual(len(kg_dict[100]), 2)

        user_ts = generate_user_triple_set(liked_items=[100], kg_dict=kg_dict, n_layer=1, set_size=16)
        self.assertEqual(len(user_ts), 1)
        h, r, t = user_ts[0]
        self.assertEqual(len(h), 16)
        self.assertTrue(all(head == 100 for head in h))
        self.assertTrue(all(rel in (0, 1) for rel in r))
        self.assertTrue(all(tail in (1001, 2001) for tail in t))

    def test_empty_user_fallback(self):
        sample_kg = np.array([[0, 0, 0]], dtype=np.int32)
        kg_dict = build_kg_dict(sample_kg)
        user_ts = generate_user_triple_set(liked_items=[], kg_dict=kg_dict, n_layer=1, set_size=8)
        self.assertEqual(len(user_ts), 1)
        self.assertEqual(len(user_ts[0][0]), 8)

    def test_multi_domain_initialization_and_recs(self):
        recommendation_engine.initialize()
        domains = recommendation_engine.get_domains_info()
        domain_ids = [d.id for d in domains]
        self.assertIn("movie", domain_ids)
        self.assertIn("book", domain_ids)
        self.assertIn("music", domain_ids)

        for d in ["movie", "book", "music"]:
            recs = recommendation_engine.recommend(d, user_id=1, top_k=3)
            self.assertGreaterEqual(len(recs), 1)
            self.assertIsNotNone(recs[0].title)

    def test_cold_start_simulation(self):
        recommendation_engine.initialize()
        sim = recommendation_engine.simulate_cold_start("movie", interactions=3)
        self.assertIsNotNone(sim.currentMetrics)
        self.assertGreater(sim.currentMetrics.ckan_auc, sim.currentMetrics.cf_auc)
        self.assertEqual(len(sim.trajectory), 6)

if __name__ == "__main__":
    unittest.main()
