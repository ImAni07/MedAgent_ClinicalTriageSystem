# test_medagent_environment.py

# Tests for the MedAgent environment implementation and API integration. 
# This includes unit tests for the environment's core logic and an integration test that verifies the WebSocket API endpoints work as expected with the environment's state management.

# Import Requirements
from __future__ import annotations
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from models import MedAgentAction, MedAgentObservation, MedAgentState
from server.app import app
from server.environment import MedAgentEnvironment

# Mock knowledge base to simulate MedQuad without heavy loading
FAKE_KNOWLEDGE = {
    "disease_to_symptoms": {
        "angina": ["chest pain", "fatigue", "shortness of breath"],
        "gastroenteritis": ["abdominal pain", "diarrhea", "vomiting"],
        "influenza": ["cough", "fever", "muscle aches"],
    },
    
    "symptom_to_diseases": {
        "abdominal pain": ["gastroenteritis"],
        "chest pain": ["angina"],
        "cough": ["influenza"],
        "diarrhea": ["gastroenteritis"],
        "fatigue": ["angina"],
        "fever": ["influenza"],
        "muscle aches": ["influenza"],
        "shortness of breath": ["angina"],
        "vomiting": ["gastroenteritis"],
    },
    
    "symptom_to_disease": {
        "abdominal pain": "gastroenteritis",
        "chest pain": "angina",
        "cough": "influenza",
        "diarrhea": "gastroenteritis",
        "fatigue": "angina",
        "fever": "influenza",
        "muscle aches": "influenza",
        "shortness of breath": "angina",
        "vomiting": "gastroenteritis",
    },
}

# Setup mock dataset and knowledge cache before each test
class MedAgentEnvironmentTests(unittest.TestCase):
    
    def setUp(self) -> None:
        
        self.knowledge_patcher = patch.object(MedAgentEnvironment, "_knowledge_cache", FAKE_KNOWLEDGE)
        self.dataset_patcher = patch.object(MedAgentEnvironment, "_dataset_cache", object())
        self.knowledge_patcher.start()
        self.dataset_patcher.start()
        self.addCleanup(self.knowledge_patcher.stop)
        self.addCleanup(self.dataset_patcher.stop)

    # Verify reset generates valid patient observation and initializes state correctly
    def test_reset_returns_patient_observation(self) -> None:
        
        env = MedAgentEnvironment()
        observation = env.reset(seed=7, episode_id="episode-123")

        self.assertIsInstance(observation, MedAgentObservation)
        self.assertGreaterEqual(observation.age, 0)
        self.assertTrue(observation.symptoms)
        self.assertEqual(observation.reward, 0.0)
        self.assertFalse(observation.done)
        self.assertEqual(env.state.episode_id, "episode-123")
        self.assertEqual(env.state.step_count, 0)
        self.assertEqual(env.state.cumulative_reward, 0.0)

    # Check that correct action yields full reward and completes episode
    def test_exact_match_returns_full_reward(self) -> None:
        env = MedAgentEnvironment()
        env.reset(seed=11)

        action = MedAgentAction(
            risk_level=env.state.target_risk,
            decision=env.state.target_decision,
        )
        
        observation = env.step(action)

        self.assertEqual(observation.reward, 1.0)
        self.assertTrue(observation.done)
        self.assertEqual(env.state.step_count, 1)
        self.assertEqual(env.state.cumulative_reward, 1.0)
        self.assertIsNotNone(observation.predicted_disease)

    # Verify partial reward when risk is correct but decision is incorrect
    def test_correct_risk_wrong_decision_gets_partial_reward(self) -> None:
        
        env = MedAgentEnvironment()
        
        env._state = MedAgentState(
            episode_id="partial",
            step_count=0,
            age=40,
            symptoms=["fever", "cough"],
            expected_disease="influenza",
            target_risk="medium",
            target_decision="book_consult",
            cumulative_reward=0.0,
        )

        observation = env.step(
            MedAgentAction(
                risk_level="medium",
                decision="monitor_at_home",
            )
        )

        self.assertGreater(observation.reward, 0.0)
        self.assertLess(observation.reward, 1.0)

    # Ensure unsafe under-triage is penalized with very low reward
    def test_unsafe_undertriage_stays_low_reward(self) -> None:
        
        env = MedAgentEnvironment()
        
        env._state = MedAgentState(
            episode_id="unsafe",
            step_count=0,
            age=72,
            symptoms=["chest pain", "shortness of breath"],
            expected_disease="angina",
            target_risk="high",
            target_decision="seek_emergency_care",
            cumulative_reward=0.0,
        )

        observation = env.step(
            MedAgentAction(
                risk_level="low",
                decision="monitor_at_home",
            )
        )

        self.assertLessEqual(observation.reward, 0.1)
        self.assertIn("under-triaged", observation.feedback)

# Optional test to verify real MedQuad dataset loading (disabled by default)
class MedAgentIntegrationTests(unittest.TestCase):
    
    @unittest.skipUnless(os.getenv('RUN_MEDQUAD_INTEGRATION') == '1', 'Enable RUN_MEDQUAD_INTEGRATION=1 to run the real MedQuad loader smoke test.')
    
    def test_medquad_cache_loader_builds_mapping(self) -> None:
        
        csv_path = MedAgentEnvironment._discover_cached_csv()
        
        if csv_path is None:
            self.skipTest("Cached MedQuad CSV not available for integration smoke test.")

        with patch.dict(os.environ, {"HF_DATASETS_OFFLINE": "1"}, clear=False):
            
            with patch.object(MedAgentEnvironment, "_dataset_cache", None):
                
                with patch.object(MedAgentEnvironment, "_knowledge_cache", None):
                    env = MedAgentEnvironment()

        self.assertTrue(env.disease_to_symptoms)
        self.assertTrue(env.symptom_to_disease)

    # Test full API interaction using WebSocket (reset → step → state)
    def test_websocket_session_reset_step_and_state(self) -> None:
        
        with patch.object(MedAgentEnvironment, "_knowledge_cache", FAKE_KNOWLEDGE):
            
            with patch.object(MedAgentEnvironment, "_dataset_cache", object()):
                client = TestClient(app)
                
                with client.websocket_connect("/ws") as websocket:
                    
                    websocket.send_json({"type": "reset", "data": {"seed": 3}})
                    reset_payload = websocket.receive_json()
                    
                    self.assertEqual(reset_payload["type"], "observation")
                    self.assertIn("symptoms", reset_payload["data"]["observation"])

                    websocket.send_json(
                        {
                            "type": "step",
                            "data": {
                                "risk_level": "medium",
                                "decision": "book_consult",
                            },
                        }
                    )
                    
                    step_payload = websocket.receive_json()
                    
                    self.assertEqual(step_payload["type"], "observation")
                    self.assertIn("reward", step_payload["data"])

                    websocket.send_json({"type": "state"})
                    state_payload = websocket.receive_json()
                    
                    self.assertEqual(state_payload["type"], "state")
                    self.assertIn("episode_id", state_payload["data"])

# Main Execution
if __name__ == "__main__":
    unittest.main()