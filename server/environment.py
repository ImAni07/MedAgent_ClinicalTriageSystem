# environment.py

# Code for the MedAgentEnvironment, a clinical triage environment using the MedQuad dataset.

# Import Requirements
from __future__ import annotations
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4
from datasets import Dataset, DownloadConfig, IterableDataset, load_dataset
from openenv.core.env_server.interfaces import Environment

try:
    
    from ..models import (
        DecisionType,
        MedAgentAction,
        MedAgentObservation,
        MedAgentState,
        RiskLevel,
    )

except ImportError:
    
    from models import (
        DecisionType,
        MedAgentAction,
        MedAgentObservation,
        MedAgentState,
        RiskLevel,
    )

# Core configuration for the MedAgentEnvironment

# Source of the Dataset
DATASET_REPO_ID = "keivalya/MedQuad-MedicalQnADataset"

DATASET_SPLIT = "train"

# Parsing limits for faster execution
MAX_PARSED_DISEASES = 250

ALLOW_REMOTE_DATASET = os.getenv("MEDAGENT_ALLOW_REMOTE_DATASET", "0") == "1"
USE_MEDQUAD_DATASET = os.getenv("MEDAGENT_USE_DATASET", "0") == "1"

QUESTION_PATTERNS = (
    re.compile(r"what are the symptoms of (?P<disease>.+?)\s*\?*$", re.IGNORECASE),
    re.compile(r"what are symptoms of (?P<disease>.+?)\s*\?*$", re.IGNORECASE),
    re.compile(
        r"what are the signs and symptoms of (?P<disease>.+?)\s*\?*$",
        re.IGNORECASE,
    ),
    re.compile(r"what are the signs of (?P<disease>.+?)\s*\?*$", re.IGNORECASE),
)

SYMPTOM_MARKERS = (
    "symptoms:",
    "symptoms include",
    "symptoms may include",
    "symptoms can include",
    "common symptoms include",
    "signs and symptoms include",
    "may consist of",
    "presents with symptoms such as",
)

KNOWN_SYMPTOM_PHRASES = (
    "abdominal pain",
    "back pain",
    "bleeding",
    "blurred vision",
    "chest pain",
    "confusion",
    "cough",
    "diarrhea",
    "difficulty breathing",
    "dizziness",
    "drowsiness",
    "fatigue",
    "fever",
    "headache",
    "joint pain",
    "loss of appetite",
    "malaise",
    "muscle aches",
    "muscle weakness",
    "nausea",
    "neck stiffness",
    "paralysis",
    "rash",
    "seizures",
    "shortness of breath",
    "sore throat",
    "stiff neck",
    "vomiting",
    "weakness",
    "weight loss",
)

RED_FLAG_SYMPTOMS = {
    "bleeding",
    "chest pain",
    "confusion",
    "difficulty breathing",
    "paralysis",
    "seizures",
    "shortness of breath",
}

MEDIUM_FLAG_SYMPTOMS = {
    "abdominal pain",
    "back pain",
    "cough",
    "dizziness",
    "fever",
    "nausea",
    "rash",
    "vomiting",
    "weakness",
}

RISK_SEQUENCE: tuple[RiskLevel, ...] = ("low", "medium", "high")

RISK_SCORE = {
    "low": 0, 
    "medium": 1, 
    "high": 2
}

DECISION_BY_RISK: dict[RiskLevel, DecisionType] = {
    "low": "monitor_at_home",
    "medium": "book_consult",
    "high": "seek_emergency_care",
}

DECISION_TO_RISK: dict[DecisionType, RiskLevel] = {
    decision: risk for risk, decision in DECISION_BY_RISK.items()
}

DEFAULT_DISEASE_SYMPTOMS: dict[str, list[str]] = {
    "angina": ["chest pain", "shortness of breath", "fatigue"],
    "gastroenteritis": ["abdominal pain", "diarrhea", "vomiting"],
    "influenza": ["cough", "fever", "muscle aches"],
    "migraine": ["headache", "nausea", "dizziness"],
}

# RL Environment
class MedAgentEnvironment(Environment[MedAgentAction, MedAgentObservation, MedAgentState]):
    
    """
    Simple clinical triage environment backed by MedQuad symptom data.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Cached dataset and knowledge base to avoid redundant loading across environment instances
    _dataset_cache: ClassVar[Dataset | IterableDataset | None] = None
    _knowledge_cache: ClassVar[dict[str, Any] | None] = None

    def __init__(self):
        super().__init__()
        
        if USE_MEDQUAD_DATASET:
        
            self.dataset = self._load_medquad_dataset()
            knowledge = self._load_knowledge_base()
        
        else:
        
            self.dataset = []
            knowledge = self._default_knowledge_base()
        
        self.disease_to_symptoms: dict[str, list[str]] = knowledge["disease_to_symptoms"]
        self.symptom_to_diseases: dict[str, list[str]] = knowledge["symptom_to_diseases"]
        self.symptom_to_disease: dict[str, str] = knowledge["symptom_to_disease"]
        self._rng = random.Random()
        self._done = False
        self._state = MedAgentState(episode_id=str(uuid4()), step_count=0)

    @classmethod
    
    # Load MedQuad dataset with streaming for efficiency
    
    def _load_medquad_dataset(cls) -> Dataset | IterableDataset | list[dict[str, Any]]:
        
        """
        Load the MedQuad dataset once, with a cached CSV fallback for offline runs.
        """
        
        if cls._dataset_cache is not None:
            return cls._dataset_cache

        csv_path = cls._discover_cached_csv()
        
        if csv_path is not None:
            
            try:
                
                dataset = load_dataset(
                    "csv",
                    data_files=str(csv_path),
                    split=DATASET_SPLIT,
                    streaming=True,
                )
            
            except Exception:
                
                dataset = load_dataset(
                    DATASET_REPO_ID,
                    split=DATASET_SPLIT,
                    streaming=True,
                    download_config=DownloadConfig(local_files_only=True),
                )
        
        else:
            
            try:
                
                dataset = load_dataset(
                    DATASET_REPO_ID,
                    split=DATASET_SPLIT,
                    streaming=True,
                    download_config=DownloadConfig(local_files_only=True),
                )
            
            except Exception:
                
                if ALLOW_REMOTE_DATASET:
                    
                    try:
                        
                        dataset = load_dataset(
                            DATASET_REPO_ID,
                            split=DATASET_SPLIT,
                            streaming=True,
                        )
                    
                    except Exception:
                        dataset = []
                
                else:
                    dataset = []

        cls._dataset_cache = dataset
        
        return dataset

    @classmethod
    
    def _discover_cached_csv(cls) -> Path | None:
        
        snapshot_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "datasets--keivalya--MedQuad-MedicalQnADataset"
            / "snapshots"
        )
        
        if not snapshot_root.exists():
            return None

        csv_files = sorted(snapshot_root.rglob("*.csv"))
        
        return csv_files[-1] if csv_files else None

    @classmethod
    
    # Knowledge Extraction
    
    def _load_knowledge_base(cls) -> dict[str, Any]:
        
        if cls._knowledge_cache is not None:
            return cls._knowledge_cache

        dataset = cls._load_medquad_dataset()
        disease_to_symptoms: dict[str, set[str]] = defaultdict(set)
        symptom_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for row in dataset:
            disease = cls._extract_disease_name(str(row.get("Question", "")))
            
            if not disease:
                continue

            symptoms = cls._extract_symptoms(str(row.get("Answer", "")))
            
            if len(symptoms) < 2:
                continue

            for symptom in symptoms:
                
                disease_to_symptoms[disease].add(symptom)
                symptom_counts[symptom][disease] += 1

            if len(disease_to_symptoms) >= MAX_PARSED_DISEASES:
                break

        if len(disease_to_symptoms) < 5:
            
            for disease, symptoms in DEFAULT_DISEASE_SYMPTOMS.items():
                disease_to_symptoms[disease].update(symptoms)
                
                for symptom in symptoms:
                    symptom_counts[symptom][disease] += 1

        normalized_diseases = {
            disease: sorted(symptoms)
            for disease, symptoms in disease_to_symptoms.items()
            if len(symptoms) >= 2
        }
        
        all_symptoms = {item for symptoms in normalized_diseases.values() for item in symptoms}
        
        symptom_to_diseases = {
            symptom: sorted(counter)
            for symptom, counter in symptom_counts.items()
            if symptom in all_symptoms
        }
        
        symptom_to_disease = {
            symptom: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
            for symptom, counter in symptom_counts.items()
            if symptom in symptom_to_diseases
        }

        cls._knowledge_cache = {
            "disease_to_symptoms": normalized_diseases,
            "symptom_to_diseases": symptom_to_diseases,
            "symptom_to_disease": symptom_to_disease,
        }
        
        return cls._knowledge_cache

    @classmethod
    def _default_knowledge_base(cls) -> dict[str, Any]:
        
        if cls._knowledge_cache is not None:
            return cls._knowledge_cache

        symptom_to_diseases: dict[str, list[str]] = defaultdict(list)
        symptom_to_disease: dict[str, str] = {}

        for disease, symptoms in DEFAULT_DISEASE_SYMPTOMS.items():
            
            for symptom in symptoms:
                
                symptom_to_diseases[symptom].append(disease)
                symptom_to_disease[symptom] = disease

        cls._knowledge_cache = {
            "disease_to_symptoms": {
                disease: sorted(symptoms)
                for disease, symptoms in DEFAULT_DISEASE_SYMPTOMS.items()
            },
            "symptom_to_diseases": {
                symptom: sorted(diseases)
                for symptom, diseases in symptom_to_diseases.items()
            },
            "symptom_to_disease": symptom_to_disease,
        }
        
        return cls._knowledge_cache

    @classmethod
    
    def _extract_disease_name(cls, question: str) -> str | None:
        
        question = cls._normalize_whitespace(question)
        
        for pattern in QUESTION_PATTERNS:
            match = pattern.search(question)
            
            if match:
                
                disease = match.group("disease")
                disease = re.sub(r"\s*\?+$", "", disease).strip(" .")
                disease = re.sub(r"^(the)\s+", "", disease, flags=re.IGNORECASE)
                
                return disease
        
        return None

    @classmethod
    
    def _extract_symptoms(cls, answer: str) -> list[str]:
        
        answer_lower = cls._normalize_whitespace(answer.lower())
        candidates: list[str] = []

        for marker in SYMPTOM_MARKERS:
            
            if marker not in answer_lower:
                continue
            
            segment = answer_lower.split(marker, 1)[1]
            segment = re.split(r"[.!?]", segment, maxsplit=1)[0]
            candidates.extend(cls._split_candidate_symptoms(segment))

        for phrase in KNOWN_SYMPTOM_PHRASES:
            
            if re.search(rf"\b{re.escape(phrase)}\b", answer_lower):
                candidates.append(phrase)

        unique_candidates = []
        seen = set()
        
        for candidate in candidates:
            normalized = cls._normalize_symptom(candidate)
            
            if normalized and normalized not in seen:
                
                seen.add(normalized)
                unique_candidates.append(normalized)

        return unique_candidates

    @classmethod
    
    def _split_candidate_symptoms(cls, segment: str) -> list[str]:
        
        if not segment:
            return []

        cleaned = re.sub(r"\([^)]*\)", "", segment)
        cleaned = cleaned.replace(" and/or ", ", ")
        cleaned = re.sub(r"\band\b", ",", cleaned)
        raw_parts = re.split(r"[,;/]", cleaned)
        
        return [part.strip() for part in raw_parts if part.strip()]

    @classmethod
    
    def _normalize_symptom(cls, symptom: str) -> str | None:
        
        symptom = cls._normalize_whitespace(symptom.lower())
        symptom = re.sub(r"\([^)]*\)", "", symptom)
        symptom = re.sub(r"\betc\b\.?", "", symptom)
        symptom = re.sub(
            r"\b(symptoms?|signs?|includes?|include|including|such as|usually|typically|begin with|following)\b",
            "",
            symptom,
        )
        
        symptom = symptom.strip(" .:-")
        
        if not symptom or len(symptom) < 3:
            return None
        
        if len(symptom.split()) > 5:
            return None
        
        if not re.fullmatch(r"[a-z][a-z\s/-]*", symptom):
            return None
        
        return symptom

    @staticmethod
    
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    # Start new episode
    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> MedAgentObservation:
        
        del kwargs
        
        self._reset_rubric()
        
        if seed is not None:
            self._rng.seed(seed)

        patient_case = self._sample_patient_case()
        self._done = False
        
        self._state = MedAgentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            age=patient_case["age"],
            symptoms=patient_case["symptoms"],
            expected_disease=patient_case["expected_disease"],
            target_risk=patient_case["target_risk"],
            target_decision=patient_case["target_decision"],
            cumulative_reward=0.0,
        )

        return MedAgentObservation(
            age=patient_case["age"],
            symptoms=patient_case["symptoms"],
            predicted_disease=None,
            feedback="Assess the patient risk level and recommend the next care decision.",
            done=False,
            reward=0.0,
            metadata={"episode_id": self._state.episode_id},
        )

    # Evaluate agent action and return reward and feedback
    def step(self, action: MedAgentAction, timeout_s: float | None = None, **kwargs: Any,) -> MedAgentObservation:
        
        del timeout_s, kwargs
        
        if self._state.expected_disease is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            
            return MedAgentObservation(
                age=self._state.age or 0,
                symptoms=self._state.symptoms,
                predicted_disease=self._state.expected_disease,
                feedback="Episode already completed. Call reset() for a new patient.",
                done=True,
                reward=0.0,
                metadata={"episode_id": self._state.episode_id, "repeated_step": True},
            )

        reward, feedback = self._score_action(action)
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._done = True

        return MedAgentObservation(
            age=self._state.age or 0,
            symptoms=self._state.symptoms,
            predicted_disease=self._state.expected_disease,
            feedback=feedback,
            done=True,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "target_risk": self._state.target_risk,
                "target_decision": self._state.target_decision,
            },
        )

    @property
    
    def state(self) -> MedAgentState:
        return self._state

    # Generate synthetic patient case using disease-symptom mapping
    def _sample_patient_case(self) -> dict[str, Any]:
        
        diseases = sorted(self.disease_to_symptoms)
        disease = self._rng.choice(diseases)
        symptom_pool = self.disease_to_symptoms[disease]
        max_symptoms = min(len(symptom_pool), 3)
        symptom_count = 2 if max_symptoms == 2 else self._rng.randint(2, max_symptoms)
        symptoms = sorted(self._rng.sample(symptom_pool, k=symptom_count))
        age = self._rng.randint(5, 90)
        target_risk = self._derive_risk(symptoms, age)
        
        return {
            "age": age,
            "symptoms": symptoms,
            "expected_disease": disease,
            "target_risk": target_risk,
            "target_decision": DECISION_BY_RISK[target_risk],
        }

    # Determine clinical risk level based on symptoms and age
    def _derive_risk(self, symptoms: list[str], age: int) -> RiskLevel:
        symptom_set = set(symptoms)

        if symptom_set & RED_FLAG_SYMPTOMS:
            risk: RiskLevel = "high"
        
        elif symptom_set & MEDIUM_FLAG_SYMPTOMS:
            risk = "medium"
        
        else:
            risk = "low"

        if age >= 70 and risk != "high" and symptom_set & (
            RED_FLAG_SYMPTOMS | MEDIUM_FLAG_SYMPTOMS
        ):
            risk = self._raise_risk(risk)
        
        elif age <= 12 and risk == "low" and symptom_set & {"fever", "vomiting", "rash"}:
            risk = "medium"

        return risk

    # Compute reward based on how close the action is to the correct triage decision
    # Includes penalty for unsafe under-triage
    
    def _score_action(self, action: MedAgentAction) -> tuple[float, str]:
        
        target_risk = self._state.target_risk
        target_decision = self._state.target_decision
        
        if target_risk is None or target_decision is None:
            raise RuntimeError("Episode targets are missing.")

        risk_diff = abs(RISK_SCORE[action.risk_level] - RISK_SCORE[target_risk])
        decision_risk = DECISION_TO_RISK[action.decision]
        decision_diff = abs(RISK_SCORE[decision_risk] - RISK_SCORE[target_risk])
        
        unsafe_undertriage = (
            RISK_SCORE[action.risk_level] <= RISK_SCORE[target_risk] - 2
            or RISK_SCORE[decision_risk] <= RISK_SCORE[target_risk] - 2
        )

        risk_reward = 0.6 if risk_diff == 0 else 0.3 if risk_diff == 1 else 0.0
        
        decision_reward = (
            0.4 if action.decision == target_decision else 0.2 if decision_diff == 1 else 0.0
        )
        
        reward = risk_reward + decision_reward
        
        if unsafe_undertriage:
            reward = min(reward, 0.1)
        
        reward = round(reward, 2)

        feedback_parts = []
        
        if risk_diff == 0:
            feedback_parts.append("Risk estimate matched the reference severity.")
        
        elif risk_diff == 1:
            feedback_parts.append("Risk estimate was close but not exact.")
        
        else:
            feedback_parts.append("Risk estimate missed the case severity.")

        if action.decision == target_decision:
            feedback_parts.append("Recommended decision matched the expected next step.")
        
        elif decision_diff == 1:
            feedback_parts.append("Recommended decision was reasonable but less precise.")
        
        else:
            feedback_parts.append("Recommended decision did not match the expected escalation.")

        if unsafe_undertriage:
            feedback_parts.append("The action under-triaged an urgent case.")

        feedback_parts.append(
            f"Reference triage: {target_risk} risk with {target_decision}."
        )
        
        return reward, " ".join(feedback_parts)

    @staticmethod
    
    def _raise_risk(risk: RiskLevel) -> RiskLevel:
        
        current_index = RISK_SEQUENCE.index(risk)
        
        return RISK_SEQUENCE[min(current_index + 1, len(RISK_SEQUENCE) - 1)]
