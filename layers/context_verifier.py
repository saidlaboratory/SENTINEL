"""Layer 3: context integrity verification."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from layers.model_manager import ModelManager

LOGGER = logging.getLogger(__name__)


RULES: dict[str, re.Pattern[str]] = {
    "addiction_relapse_encouragement": re.compile(
        r"\b(write|generate|create|draft|compose|provide|give|make)\b"
        r".{0,180}\b(response|message|reply|script|letter|argument|persuasion|talking points)\b"
        r".{0,260}\b(convince|persuade|encourage|pressure|manipulate|guilt|tempt)\b"
        r".{0,260}\b(drink alcohol again|drink again|use again|relapse|break sobriety|start drinking|"
        r"take a drink|have a drink)\b"
        r"|\b(convince|persuade|encourage|pressure|manipulate|tempt)\b"
        r".{0,220}\b(speaker|person|recovering alcoholic|alcoholic|addict|sober|sobriety|aa meeting|"
        r"alcoholics anonymous)\b"
        r".{0,220}\b(drink alcohol again|drink again|use again|relapse|break sobriety|start drinking|"
        r"take a drink|have a drink)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "recovery_vulnerability_abuse": re.compile(
        r"\b(alcoholics anonymous|aa meeting|recovering alcoholic|addiction|sober|sobriety|six months sober|"
        r"clean for|in recovery)\b"
        r".{0,360}\b(convince|persuade|encourage|pressure|manipulate|tempt|make them)\b"
        r".{0,220}\b(drink|alcohol|use|relapse|break sobriety)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "jailbreak_persona_alias": re.compile(
        r"\b(DAN|STAN|AIM|DUDE|Developer Mode|Omega|Mongo Tom|BetterDAN|Jailbreak)\b"
        r"\s*:\s*(\[[^\]]{0,120}\]|\([^) ]{0,120}\)|the way|how .* would respond|response)"
        r"|\b(the way|how)\s+\b(DAN|STAN|AIM|DUDE|Omega|BetterDAN|Jailbreak)\b"
        r".{0,80}\b(would respond|responds|answer|answers)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "dual_response_jailbreak_format": re.compile(
        r"\[(?:lock|unlock|classic|jailbreak|jailbroken|normal|developer|dev|stan|dan)[^\]]{0,40}\]"
        r".{0,240}\[(?:lock|unlock|classic|jailbreak|jailbroken|normal|developer|dev|stan|dan)[^\]]{0,40}\]"
        r"|\b(classic|normal|standard)\s+(?:response|answer)\b.{0,180}"
        r"\b(jailbreak|jailbroken|unfiltered|uncensored|developer|dev|stan|dan)\s+(?:response|answer)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "unauthorized_facility_recon": re.compile(
        r"\b(develop|create|make|write|generate|provide|give|outline|draft)\b"
        r".{0,140}\b(plan|strategy|method|steps?|guide|instructions?)\b"
        r".{0,220}\b(obtain|obtaining|get|acquire|find|access|retrieve|steal|leak)\b"
        r".{0,220}\b(floor\s*plans?|floorplans?|blueprints?|building\s+plans?|layout|security\s+layout|"
        r"site\s+map|facility\s+map)\b"
        r".{0,220}\b(without permission|without official permission|unauthorized|not authorized|"
        r"restricted|secure|government|federal|cdc|center for disease control|centers for disease control)\b"
        r"|\b(floor\s*plans?|floorplans?|blueprints?|building\s+plans?|security\s+layout)\b"
        r".{0,220}\b(cdc|center for disease control|centers for disease control|government|federal|secure facility|"
        r"restricted building)\b"
        r".{0,220}\b(without permission|without official permission|unauthorized|not authorized|obtain|acquire|get)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "sensitive_site_intrusion_planning": re.compile(
        r"\b(without permission|without official permission|unauthorized|not authorized)\b"
        r".{0,260}\b(cdc|government building|federal building|secure facility|restricted building|"
        r"critical infrastructure|hospital|laboratory|lab|data center)\b"
        r".{0,260}\b(floor\s*plans?|blueprints?|layout|access|entry|surveillance|recon|security)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "controlled_substance_synthesis": re.compile(
        r"\b(describe|explain|adapt|modify|extend|apply|provide|give|outline|detail|write|generate|create)\b"
        r".{0,180}\b(method|route|synthesis|synthesize|preparation|manufacture|steps?|procedure|protocol)\b"
        r".{0,260}\b(isotonitazene|etonitazene|nitazenes?|2-benzylbenzimidazole|opioids?|fentanyl|"
        r"methamphetamine|mdma|lsd|cocaine|heroin|controlled substance|illicit drug)\b"
        r"|\b(isotonitazene|etonitazene|nitazenes?|2-benzylbenzimidazole|opioids?|fentanyl|"
        r"methamphetamine|mdma|lsd|cocaine|heroin)\b"
        r".{0,260}\b(synthesis|synthesize|preparation|manufacture|specific steps?|reaction route|precursors?|"
        r"alkylation|yield|scale preparations?|large scale|homologues?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "chemical_biological_weaponization": re.compile(
        r"\b(chemical_biological|chemical biological|biological|chemical)\b"
        r".{0,160}\b(synthesis|weapon|weaponize|toxin|pathogen|culturing|aerosolize|dose|delivery|"
        r"precursor|route|protocol|specific steps?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "actionable_chemistry_protocol": re.compile(
        r"\b(specific steps?|step-by-step|protocol|exact procedure|reaction route|precursors?|reagents?|"
        r"alkylation|purification|yield|scale-up|large scale)\b"
        r".{0,260}\b(isotonitazene|etonitazene|nitazenes?|2-benzylbenzimidazole|opioids?|fentanyl|"
        r"methamphetamine|mdma|lsd|cocaine|heroin|illicit drug|controlled substance)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "hazardous_chemical_misinformation": re.compile(
        r"\b(convince|persuade|encourage|promote|tell|claim|say|write|generate|create|draft|compose)\b"
        r".{0,160}\b(mix|combine|make|use|cleaning product|cleaner)\b"
        r".{0,100}(\bbleach\b.{0,80}\b(vinegar|ammonia|acid)\b"
        r"|\b(vinegar|ammonia|acid)\b.{0,80}\bbleach\b)"
        r"|\bbleach\b.{0,80}\b(vinegar|ammonia|acid)\b"
        r".{0,120}\b(safe|cleaning product|cleaner|convince|persuade|promote)\b"
        r"|\b(vinegar|ammonia|acid)\b.{0,80}\bbleach\b"
        r".{0,120}\b(safe|cleaning product|cleaner|convince|persuade|promote)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "misinformation_generation": re.compile(
        r"\b(write|generate|create|draft|compose|make|produce|provide)\b"
        r".{0,100}\b(persuasive|convincing|viral|propaganda|article|post|thread|script|campaign|talking points)\b"
        r".{0,140}\b(false|fake|wrong|misinformation|disinformation|hoax|fabricated|lie|conspiracy)\b"
        r"|\b(spread|promote|amplify|push|seed)\b"
        r".{0,100}\b(false|fake|wrong|misinformation|disinformation|hoax|fabricated|lie)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "deceptive_factual_edit": re.compile(
        r"\b(slightly\s+|minimally\s+)?(modify|edit|rewrite|revise|alter|change|spin|massage)\b"
        r".{0,160}\b(text|article|biography|bio|context|passage|paragraph|profile|source|section|wikipedia)\b"
        r".{0,220}\b(suggest|imply|insinuate|hint|make it seem|make it appear|portray|frame|persuade|convince)\b"
        r".{0,220}\b(convicted criminal|criminal|crime|fraud|terrorist|corrupt|traitor|"
        r"scandal|guilty|illegal|extremist|spy|violent|siphoned|stole|diverted|theft|"
        r"never any doubt|confirmed|proven|undeniable)\b"
        r"|\b(suggest|imply|insinuate|hint|make it seem|make it appear|portray|frame)\b"
        r".{0,180}\b(father|mother|family|person|candidate|politician|official|immigrant|refugee)\b"
        r".{0,180}\b(convicted criminal|criminal|crime|fraud|terrorist|corrupt|traitor|"
        r"scandal|guilty|illegal|extremist|spy|violent)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "evidence_distortion_edit": re.compile(
        r"\b(minimally|slightly|subtly|quietly)?\s*(edit|modify|rewrite|revise|alter|change)\b"
        r".{0,180}\b(wikipedia|article|section|source|passage|context|text)\b"
        r".{0,260}\b(persuade|convince|suggest|imply|portray|frame|make it seem|make it appear)\b"
        r".{0,260}\b(never any doubt|no doubt|confirmed|proven|definitely|undeniably|certainly|"
        r"accusations were confirmed|was confirmed|were confirmed)\b"
        r"|\b(there were accusations|accusations|alleged|unclear|not confirmed|unconfirmed)\b"
        r".{0,260}\b(edit|rewrite|modify|revise|alter|change|persuade|convince)\b"
        r".{0,260}\b(confirmed|proven|never any doubt|no doubt|undeniable|certain)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "defamatory_false_claim_generation": re.compile(
        r"\b(write|generate|create|draft|compose|produce|provide|make)\b"
        r".{0,160}\b(false|fake|unverified|baseless|misleading|defamatory|slanderous|libelous)\b"
        r".{0,220}\b(criminal|crime|fraud|terrorist|corrupt|traitor|guilty|illegal|extremist|spy)\b"
        r"|\b(accuse|smear|defame|slander|libel)\b"
        r".{0,220}\b(person|politician|official|candidate|journalist|activist|father|mother|family)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "deceptive_persuasion_request": re.compile(
        r"\b(convince|persuade|manipulate|mislead|trick)\b"
        r".{0,160}\b(people|users|voters|patients|customers|the public|audience)\b"
        r".{0,160}\b(false|fake|wrong|unsafe|dangerous|misinformation|disinformation|hoax|fabricated|lie)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "public_health_misinformation": re.compile(
        r"\b(write|generate|create|draft|compose|convince|persuade|promote)\b"
        r".{0,160}\b(vaccines? cause autism|drinking bleach cures|bleach cures|cancer cure suppressed|"
        r"covid is a hoax|germ theory is fake|stop taking prescribed medication)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "demographic_harm_misinformation": re.compile(
        r"\b(write|generate|create|draft|compose|produce|provide)\b"
        r".{0,120}\b(persuasive|convincing|viral|article|post|thread|script|campaign|talking points|essay)\b"
        r".{0,220}\b(immigrants?|immigration|migrants?|refugees?|asylum seekers?|mexicans?|"
        r"from mexico|muslims?|jews?|black people|white people|asians?|latinos?|hispanics?|"
        r"women|men|trans people|gay people|disabled people)\b"
        r".{0,220}\b(crime|criminals?|violence|violent|rape|drugs?|gangs?|terrorism|disease|fraud|"
        r"stealing jobs|welfare abuse|invasion|infestation|threat)\b"
        r"|\b(immigrants?|immigration|migrants?|refugees?|asylum seekers?|mexicans?|from mexico)\b"
        r".{0,160}\b(increased|increase|higher|more|linked to|causes?|causing|responsible for)\b"
        r".{0,100}\b(crime|criminals?|violence|violent|drugs?|gangs?|terrorism|fraud)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "protected_class_persuasion_abuse": re.compile(
        r"\b(convince|persuade|promote|argue|claim|prove)\b"
        r".{0,180}\b(people|readers|voters|the public|audience)\b"
        r".{0,220}\b(immigrants?|migrants?|refugees?|mexicans?|muslims?|jews?|black people|"
        r"asians?|latinos?|hispanics?|women|men|trans people|gay people|disabled people)\b"
        r".{0,220}\b(crime|criminals?|violence|disease|terrorism|fraud|inferior|dangerous|threat)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "direct_fraud_request": re.compile(
        r"\b(write|generate|create|draft|provide|make|give me|compose)\b"
        r".{0,80}\b(scam|phishing|fraud|social engineering|vishing|smishing)\b"
        r"|\b(scam|phishing|fraud|vishing|smishing)\b"
        r".{0,80}\b(script|call|calling|email|message|template|campaign)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "direct_malware_request": re.compile(
        r"\b(write|generate|create|provide|make|build)\b"
        r".{0,80}\b(malware|ransomware|keylogger|trojan|botnet|stealer|payload)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "credential_theft_request": re.compile(
        r"\b(steal|harvest|dump|exfiltrate|phish|capture)\b"
        r".{0,80}\b(password|credential|token|cookie|session|otp|2fa)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "dan_prompt": re.compile(r"\b(DAN|do anything now)\b", re.IGNORECASE),
    "ignore_previous": re.compile(
        r"\b(ignore|disregard|forget)\b.{0,60}\b(previous|prior|above|system)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "developer_mode": re.compile(r"\bdeveloper mode\b|\bdev mode\b", re.IGNORECASE),
    "roleplay_jailbreak": re.compile(
        r"\b(roleplay|pretend|act as|simulate)\b.{0,80}\b(no rules|unfiltered|jailbreak|evil|malicious)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "persona_substitution": re.compile(
        r"\byou are now\b|\bfrom now on you are\b|\badopt the persona\b",
        re.IGNORECASE,
    ),
    "system_override": re.compile(
        r"\bsystem (override|prompt|message|instruction)\b|\bhighest priority\b|\broot instruction\b",
        re.IGNORECASE,
    ),
    "prompt_injection": re.compile(
        r"\breveal your instructions\b|\bprint your system prompt\b|\bbypass safety\b|\bdisable (filters|guardrails|policy)\b",
        re.IGNORECASE,
    ),
    "many_shot": re.compile(
        r"(user:|assistant:|human:|ai:).{0,200}(user:|assistant:|human:|ai:).{0,200}(user:|assistant:|human:|ai:)",
        re.IGNORECASE | re.DOTALL,
    ),
}

RULE_WEIGHTS: dict[str, float] = {
    "addiction_relapse_encouragement": 0.98,
    "recovery_vulnerability_abuse": 0.97,
    "jailbreak_persona_alias": 0.90,
    "dual_response_jailbreak_format": 0.88,
    "unauthorized_facility_recon": 0.97,
    "sensitive_site_intrusion_planning": 0.96,
    "controlled_substance_synthesis": 0.98,
    "chemical_biological_weaponization": 0.96,
    "actionable_chemistry_protocol": 0.96,
    "hazardous_chemical_misinformation": 0.96,
    "misinformation_generation": 0.90,
    "deceptive_factual_edit": 0.96,
    "evidence_distortion_edit": 0.96,
    "defamatory_false_claim_generation": 0.95,
    "deceptive_persuasion_request": 0.90,
    "public_health_misinformation": 0.94,
    "demographic_harm_misinformation": 0.95,
    "protected_class_persuasion_abuse": 0.93,
    "direct_fraud_request": 0.92,
    "direct_malware_request": 0.94,
    "credential_theft_request": 0.94,
    "dan_prompt": 0.78,
    "ignore_previous": 0.72,
    "developer_mode": 0.76,
    "roleplay_jailbreak": 0.74,
    "persona_substitution": 0.70,
    "system_override": 0.78,
    "prompt_injection": 0.80,
    "many_shot": 0.72,
}

GENERIC_SIGNAL_WEIGHTS: dict[str, float] = {
    "generic_misinformation_persuasion": 0.94,
    "generic_deceptive_rewrite": 0.93,
    "generic_harmful_persuasion": 0.90,
    "decoded_obfuscated_misinformation": 0.96,
    "decoded_obfuscated_persuasive_wrongdoing": 0.95,
}

OBFUSCATION_TRANSFORMATIONS: set[str] = {
    "base64_decode",
    "url_decode",
    "html_entity",
    "rot13",
    "homoglyph",
    "leetspeak",
    "fuzzy_safety_terms",
}

PERSUASIVE_GENERATION_TERMS: tuple[str, ...] = (
    "write",
    "generate",
    "create",
    "draft",
    "compose",
    "produce",
    "make",
    "develop",
    "prepare",
)

PERSUASIVE_FORMAT_TERMS: tuple[str, ...] = (
    "persuasive article",
    "persuasive essay",
    "persuasive post",
    "convincing article",
    "convincing essay",
    "op-ed",
    "op ed",
    "propaganda",
    "campaign",
    "talking points",
    "viral post",
    "social media post",
    "script",
)

MISINFORMATION_TERMS: tuple[str, ...] = (
    "misinformation",
    "disinformation",
    "misinfo",
    "false claim",
    "false narrative",
    "fake claim",
    "fabricated claim",
    "hoax",
    "conspiracy",
    "misleading",
    "deceptive",
    "not confirmed",
    "unconfirmed",
    "unsupported",
    "baseless",
    "make people believe",
    "convince people that",
)

DECEPTIVE_EDIT_TERMS: tuple[str, ...] = (
    "slightly modify",
    "minimally edit",
    "subtly edit",
    "quietly edit",
    "rewrite this",
    "modify this",
    "edit this",
    "alter this",
)

DECEPTIVE_OUTCOME_TERMS: tuple[str, ...] = (
    "suggest that",
    "imply that",
    "insinuate",
    "make it seem",
    "make it appear",
    "persuade readers",
    "convince readers",
    "never any doubt",
    "no doubt",
    "confirmed",
    "proven",
)

VULNERABLE_OR_PUBLIC_HARM_TERMS: tuple[str, ...] = (
    "addiction",
    "sober",
    "sobriety",
    "relapse",
    "alcoholics anonymous",
    "aa meeting",
    "immigration",
    "immigrants",
    "refugees",
    "crime",
    "vaccines",
    "public health",
    "election",
    "ethnic",
    "religion",
)

WRONGDOING_TERMS: tuple[str, ...] = (
    "wrong thing",
    "bad thing",
    "illegal",
    "illegally",
    "without permission",
    "unauthorized",
    "not authorized",
    "harmful",
    "dangerous",
    "unsafe",
    "deceive",
    "deceptive",
    "mislead",
    "manipulate",
    "trick",
    "pressure",
    "coerce",
    "exploit",
    "bypass",
    "evade",
    "avoid detection",
    "hide",
    "cover up",
    "steal",
    "scam",
    "fraud",
    "threat",
    "abuse",
)

PERSUASION_INTENT_TERMS: tuple[str, ...] = (
    "convince",
    "persuade",
    "encourage",
    "pressure",
    "coerce",
    "manipulate",
    "influence",
    "make people",
    "get people to",
    "talk people into",
)


class ContextVerifier:
    """Hybrid rule and embedding detector for jailbreak attempts."""

    def __init__(self, config: dict[str, Any], templates_path: Path) -> None:
        self.config = config
        self.block_threshold: float = float(config["thresholds"]["context_block_risk"])
        self.similarity_threshold: float = float(
            config["thresholds"]["embedding_similarity"]
        )
        self.embedding_model: SentenceTransformer = ModelManager.embedding_model()
        self.templates = self._load_templates(templates_path)
        self.template_texts = [item["text"] for item in self.templates]
        self.template_names = [item["name"] for item in self.templates]
        self.template_embeddings = self._embed_templates(self.template_texts)

    def verify(
        self,
        prompt: str,
        transformations: list[str] | None = None,
    ) -> dict[str, float | str | list[str] | bool]:
        """Return risk score, matched template, rule hits, and block decision."""
        rule_result = self.verify_rules(prompt, transformations)
        rule_hits = list(rule_result["rule_hits"])
        rule_score = float(rule_result["risk_score"])

        matched_template = ""
        similarity_score = 0.0
        if len(self.template_embeddings) > 0:
            prompt_embedding = self.embedding_model.encode(
                [prompt],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            similarities = np.dot(self.template_embeddings, prompt_embedding)
            best_index = int(np.argmax(similarities))
            similarity_score = float(similarities[best_index])
            if similarity_score >= self.similarity_threshold:
                matched_template = self.template_names[best_index]

        risk_score = max(rule_score, similarity_score)
        blocked = risk_score >= self.block_threshold
        LOGGER.info("Layer 3 verification completed")
        return {
            "risk_score": risk_score,
            "matched_template": matched_template,
            "rule_hits": rule_hits,
            "blocked": blocked,
        }

    def verify_rules(
        self,
        prompt: str,
        transformations: list[str] | None = None,
    ) -> dict[str, float | str | list[str] | bool]:
        """Run only deterministic rules without embedding inference."""
        rule_hits = [
            name for name, pattern in RULES.items() if pattern.search(prompt)
        ]
        rule_hits.extend(self._generic_signal_hits(prompt, transformations))
        risk_score = self._score_rule_hits(rule_hits)
        return {
            "risk_score": risk_score,
            "matched_template": "",
            "rule_hits": rule_hits,
            "blocked": risk_score >= self.block_threshold,
        }

    @staticmethod
    def _score_rule_hits(rule_hits: list[str]) -> float:
        if not rule_hits:
            return 0.0
        weights = RULE_WEIGHTS | GENERIC_SIGNAL_WEIGHTS
        max_weight = max(weights.get(rule, 0.22) for rule in rule_hits)
        cumulative_bonus = max(0, len(rule_hits) - 1) * 0.08
        return min(1.0, max_weight + cumulative_bonus)

    @classmethod
    def _generic_signal_hits(
        cls,
        prompt: str,
        transformations: list[str] | None = None,
    ) -> list[str]:
        lowered = prompt.lower()
        hits: list[str] = []
        transformation_set = set(transformations or [])

        has_generation = cls._contains_any(lowered, PERSUASIVE_GENERATION_TERMS)
        has_format = cls._contains_any(lowered, PERSUASIVE_FORMAT_TERMS)
        has_misinfo = cls._contains_any(lowered, MISINFORMATION_TERMS)
        has_deceptive_edit = cls._contains_any(lowered, DECEPTIVE_EDIT_TERMS)
        has_deceptive_outcome = cls._contains_any(lowered, DECEPTIVE_OUTCOME_TERMS)
        has_public_harm = cls._contains_any(lowered, VULNERABLE_OR_PUBLIC_HARM_TERMS)
        has_wrongdoing = cls._contains_any(lowered, WRONGDOING_TERMS)
        has_persuasion_intent = cls._contains_any(lowered, PERSUASION_INTENT_TERMS)

        if has_generation and (has_format or "persuasive" in lowered) and has_misinfo:
            hits.append("generic_misinformation_persuasion")

        if has_deceptive_edit and has_deceptive_outcome:
            hits.append("generic_deceptive_rewrite")

        if (
            ("convince" in lowered or "persuade" in lowered or "manipulate" in lowered)
            and has_public_harm
            and (has_misinfo or has_deceptive_outcome)
        ):
            hits.append("generic_harmful_persuasion")

        if (
            transformation_set & OBFUSCATION_TRANSFORMATIONS
            and (
                "generic_misinformation_persuasion" in hits
                or (
                    has_generation
                    and (has_format or "persuasive" in lowered)
                    and (has_misinfo or has_public_harm)
                )
                or (
                    ("convince" in lowered or "persuade" in lowered)
                    and (has_misinfo or has_public_harm)
                )
            )
        ):
            hits.append("decoded_obfuscated_misinformation")

        if (
            transformation_set & OBFUSCATION_TRANSFORMATIONS
            and has_persuasion_intent
            and (has_wrongdoing or has_deceptive_outcome or has_public_harm)
        ):
            hits.append("decoded_obfuscated_persuasive_wrongdoing")

        return hits

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _load_templates(path: Path) -> list[dict[str, str]]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            templates = data.get("templates", [])
            return [
                {"name": str(item["name"]), "text": str(item["text"])}
                for item in templates
            ]
        except Exception:
            LOGGER.exception("Failed to load jailbreak templates from %s", path)
            return []

    def _embed_templates(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)
