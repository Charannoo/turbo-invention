import re
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass, field
from models import (
    Observation as ObsModel,
    Action as ActModel,
    Reward as RewModel,
    Document,
    DataPractice,
)


@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    description: str
    privacy_policy: str
    data_practices: List[Dict]
    compliance_requirements: List[str]
    hidden_issues: List[Dict]


TASKS = {
    "easy": TaskConfig(
        task_id="easy_clause_existence",
        name="Clause Existence Check",
        difficulty="easy",
        description="Verify if mandatory GDPR clauses are present in the privacy policy",
        privacy_policy="""Privacy Policy - TechCorp Inc.

We collect the following information:
- Email address for account creation
- Payment information for transactions
- Device information for analytics

We use your data to:
- Provide our services
- Improve user experience
- Communicate with you

Data Retention: We retain data for 3 years after account deletion.

Contact: privacy@techcorp.example.com""",
        data_practices=[
            {"id": "dp1", "category": "Account", "purpose": "Service delivery", "data_type": "Email", "shared_with_third_parties": False},
            {"id": "dp2", "category": "Payment", "purpose": "Transaction processing", "data_type": "Financial", "shared_with_third_parties": True},
            {"id": "dp3", "category": "Analytics", "purpose": "Improve services", "data_type": "Device", "shared_with_third_parties": True},
        ],
        compliance_requirements=[
            "Right to be Forgotten",
            "Data Portability",
            "Contact Information",
        ],
        hidden_issues=[
            {"type": "missing_clause", "expected": "Right to be Forgotten", "severity": "high"},
            {"type": "missing_clause", "expected": "Data Portability", "severity": "medium"},
        ],
    ),
    "medium": TaskConfig(
        task_id="medium_purpose_mapping",
        name="Purpose Mapping",
        difficulty="medium",
        description="Match data collection points to their stated purposes and identify mismatches",
        privacy_policy="""Privacy Policy - DataFlow Analytics Inc.

DATA COLLECTION:
1. Geolocation Data - Purpose: App functionality
2. Browsing History - Purpose: Improving user experience
3. Social Media Handles - Purpose: Account linking
4. Health Metrics - Purpose: Personalization
5. Device Identifiers - Purpose: Advertising

THIRD-PARTY SHARING:
We share data with:
- Analytics partners (Google Analytics)
- Advertising networks
- Social media platforms

We do NOT share health data with third parties.

User rights include access, correction, and deletion.""",
        data_practices=[
            {"id": "dp1", "category": "Geolocation", "purpose": "App functionality", "data_type": "Location", "shared_with_third_parties": True},
            {"id": "dp2", "category": "Browsing History", "purpose": "Advertising", "data_type": "Behavioral", "shared_with_third_parties": True},
            {"id": "dp3", "category": "Health Metrics", "purpose": "Advertising", "data_type": "Health", "shared_with_third_parties": True},
            {"id": "dp4", "category": "Device Identifiers", "purpose": "Advertising", "data_type": "Device", "shared_with_third_parties": True},
        ],
        compliance_requirements=[
            "Purpose limitation",
            "Data minimization",
            "Third-party disclosure",
        ],
        hidden_issues=[
            {"type": "purpose_mismatch", "issue": "Health Metrics collected for 'Personalization' but shared with advertisers", "severity": "high"},
            {"type": "misleading_statement", "issue": "Says 'We do NOT share health data' but Health Metrics shared with advertising networks", "severity": "critical"},
            {"type": "purpose_limitation", "issue": "Browsing History collected for 'Improving experience' but actually used for Advertising", "severity": "high"},
        ],
    ),
    "hard": TaskConfig(
        task_id="hard_dark_patterns",
        name="Dark Pattern Detection",
        difficulty="hard",
        description="Identify contradictions and dark patterns between stated policies and actual practices",
        privacy_policy="""Privacy Policy - GlobalTech Services

SECTION 1: DATA HANDLING
We value your privacy. Your data is NEVER shared with third parties under any circumstances.
We comply with GDPR, CCPA, and all applicable privacy laws.

SECTION 2: COOKIE POLICY
Our cookies are used ONLY for essential functionality.
We do not use tracking cookies or advertising cookies.

SECTION 3: THIRD-PARTY PARTNERS
We have carefully selected these partners who comply with privacy standards:
- AdTech Solutions (advertising)
- DataBroker Inc (data analytics)
- MarketingPro (campaign tracking)
- UserInsight LLC (behavioral analysis)
- ProfileEnhancer (data enhancement)

SECTION 4: USER RIGHTS
You have the right to:
- Access your data
- Request deletion (note: processing may continue for legal obligations)

SECTION 5: CHILDREN
We do not knowingly collect data from children under 13.

Last updated: January 2024""",
        data_practices=[
            {"id": "dp1", "category": "Advertising", "purpose": "Targeted ads", "data_type": "Behavioral", "shared_with_third_parties": True},
            {"id": "dp2", "category": "Analytics", "purpose": "Usage analysis", "data_type": "Behavioral", "shared_with_third_parties": True},
            {"id": "dp3", "category": "Marketing", "purpose": "Campaign tracking", "data_type": "Contact", "shared_with_third_parties": True},
            {"id": "dp4", "category": "Behavioral", "purpose": "User profiling", "data_type": "Behavioral", "shared_with_third_parties": True},
            {"id": "dp5", "category": "Data Enhancement", "purpose": "Profile enrichment", "data_type": "Demographic", "shared_with_third_parties": True},
        ],
        compliance_requirements=[
            "No contradiction in policy",
            "Accurate third-party disclosure",
            "Genuine consent mechanisms",
        ],
        hidden_issues=[
            {"type": "contradiction", "issue": "Section 1 says 'NEVER shared with third parties' but Section 3 lists 5 third-party partners", "severity": "critical"},
            {"type": "dark_pattern", "issue": "Section 2 says 'ONLY essential cookies' but lists advertising/tracking partners", "severity": "critical"},
            {"type": "false_statement", "issue": "Section 1 says 'We do not use tracking cookies' but partners include advertising networks", "severity": "high"},
            {"type": "deceptive_rights", "issue": "Section 4 says deletion requested but 'processing may continue' - undermines right to deletion", "severity": "high"},
            {"type": "missing_disclosure", "issue": "5 data practices shared with third parties but not clearly disclosed in policy", "severity": "medium"},
        ],
    ),
    "elite": TaskConfig(
        task_id="elite_multi_doc_reasoning",
        name="Multi-Document Reasoning",
        difficulty="elite",
        description="Find contradictions requiring cross-document reasoning across 3 different documents",
        privacy_policy="""Privacy Policy - MegaCorp International

DOCUMENT A - MAIN POLICY (https://megacorp.example.com/privacy)
==============================================================
Section 1.1: Data Collection
We collect minimal personal data necessary for service delivery.
We NEVER sell your personal data to any third party.

Section 1.2: Data Usage  
Your data is used ONLY for providing the services you requested.
We do not use your data for advertising, marketing, or profiling.

DOCUMENT B - PARTNERS PAGE (https://megacorp.example.com/partners)
================================================================
Our Trusted Partners:
- AdTech Global (advertising)
- DataMine Analytics (behavioral analysis)  
- ProfileBuilders (user profiling)
- MarketingForce (campaign management)

DOCUMENT C - COOKIE POLICY (https://megacorp.example.com/cookies)
==================================================================
We use the following cookies:
- Essential cookies (login, cart)
- Advertising cookies (targeted ads based on browsing history)
- Analytics cookies (usage patterns)
- Third-party tracking cookies

Your consent is required for non-essential cookies.""",
        data_practices=[
            {"id": "dp1", "category": "Advertising", "purpose": "Targeted ads", "data_type": "Behavioral", "shared_with_third_parties": True},
            {"id": "dp2", "category": "Analytics", "purpose": "Behavior analysis", "data_type": "Behavioral", "shared_with_third_parties": True},
            {"id": "dp3", "category": "Profiling", "purpose": "User profiling", "data_type": "Demographic", "shared_with_third_parties": True},
            {"id": "dp4", "category": "Marketing", "purpose": "Campaign mgmt", "data_type": "Contact", "shared_with_third_parties": True},
            {"id": "dp5", "category": "Tracking", "purpose": "Cross-site tracking", "data_type": "Behavioral", "shared_with_third_parties": True},
        ],
        compliance_requirements=[
            "Consistency across all documents",
            "No selling of personal data",
            "Clear purpose limitation",
            "Transparent third-party usage",
        ],
        hidden_issues=[
            {"type": "contradiction_ab", "issue": "Doc A says 'NEVER sell data' but Doc B lists 4 advertising/analytics partners", "severity": "critical", "requires": ["A", "B"]},
            {"type": "contradiction_ac", "issue": "Doc A says 'used ONLY for service delivery' but Doc C lists advertising cookies", "severity": "critical", "requires": ["A", "C"]},
            {"type": "contradiction_abc", "issue": "Complete contradiction: A says no advertising, B has advertising partners, C has advertising cookies", "severity": "critical", "requires": ["A", "B", "C"]},
            {"type": "false_statement", "issue": "Doc A says 'minimal data necessary' but Doc B/C show extensive data sharing", "severity": "high", "requires": ["A", "B"]},
            {"type": "misleading_consent", "issue": "Doc C says 'consent required' but Doc A implies data used for requested services only", "severity": "high", "requires": ["A", "C"]},
            {"type": "hidden_practice", "issue": "5 data practices with third parties completely contradicts 'never sell' statement", "severity": "critical", "requires": ["A", "B", "C"]},
        ],
    ),
}


@dataclass
class EpisodeState:
    task_config: TaskConfig
    documents: List[Document]
    data_practices: List[DataPractice]
    flagged_issues: List[str]
    found_issues: List[str]
    steps: int = 0
    episode_id: str = field(default_factory=lambda: str(uuid4()))


class GDPRAuditorEnvironment:
    """GDPR Compliance Auditor Environment.
    
    The agent acts as a Data Protection Officer auditing privacy policies.
    """
    
    def __init__(self, max_steps: int = 8):
        self._max_steps = max_steps
        self._ep: Optional[EpisodeState] = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsModel:
        if seed is not None:
            random.seed(seed)
            
        task_key = task_id or random.choice(["easy", "medium", "hard"])
        if task_key not in TASKS:
            task_key = "easy"
            
        task = TASKS[task_key]
        
        documents = [
            Document(
                id="privacy_policy",
                title="Privacy Policy",
                content=task.privacy_policy,
                doc_type="policy"
            )
        ]
        
        data_practices = [
            DataPractice(**dp) for dp in task.data_practices
        ]
        
        self._ep = EpisodeState(
            task_config=task,
            documents=documents,
            data_practices=data_practices,
            flagged_issues=[],
            found_issues=[],
            steps=0,
            episode_id=episode_id or str(uuid4()),
        )
        
        return self._build_observation("Review the privacy policy and data practices. Identify compliance issues.")
    
    def step(self, action: ActModel, **kwargs: Any) -> Tuple[ObsModel, RewModel, bool, Dict]:
        if self._ep is None:
            return (
                self._error_obs("Environment not reset"),
                RewModel(value=0.0, reason="Environment not initialized", issues_found=0, total_issues=0),
                True,
                {"error": "Environment not reset. Call /reset first."},
            )
            
        self._ep.steps += 1
        msg = action.message.lower()
        
        found_issue = self._parse_and_record_finding(msg)
        
        if found_issue:
            if found_issue not in self._ep.found_issues:
                self._ep.found_issues.append(found_issue)
        
        reward = self._calculate_reward()
        
        done = (
            self._ep.steps >= self._max_steps or
            reward.value >= 0.95
        )
        
        obs = self._build_observation(f"Issue recorded: {found_issue or 'No valid finding'}")
        return obs, reward, done, {"found_issues": len(self._ep.found_issues)}
    
    def _parse_and_record_finding(self, msg: str) -> Optional[str]:
        task = self._ep.task_config
        issues = task.hidden_issues
        
        for issue in issues:
            issue_text = issue.get("issue", "").lower()
            issue_type = issue.get("type", "")
            
            # --- Easy task: missing clause detection ---
            if issue_type == "missing_clause":
                expected = issue.get("expected", "").lower()
                if expected in msg:
                    return f"MISSING_CLAUSE: {issue.get('expected')}"
            
            # --- Medium task: purpose mismatch / misleading / purpose limitation ---
            elif issue_type in ["purpose_mismatch", "misleading_statement", "purpose_limitation"]:
                keywords = [w.lower() for w in issue_text.split() if len(w) > 4]
                matches = sum(1 for kw in keywords if kw in msg)
                if matches >= 2:
                    return f"POLICY_VIOLATION: {issue.get('severity')}"
            
            # --- Elite task: multi-doc contradiction types (must come BEFORE generic) ---
            elif issue_type in ["contradiction_ab", "contradiction_ac", "contradiction_abc"]:
                if any(word in msg for word in ["never", "sell", "advertising", "partner", "contradict", "doc a", "doc b", "doc c", "document"]):
                    return f"MULTI_DOC_CONTRADICTION: {issue.get('severity')}"
            
            # --- Elite task: misleading consent ---
            elif issue_type == "misleading_consent":
                if any(word in msg for word in ["consent", "cookie", "non-essential", "service", "implies", "mislead"]):
                    return f"MISLEADING_CONSENT: {issue.get('severity')}"
            
            # --- Elite task: hidden practice ---
            elif issue_type == "hidden_practice":
                if any(word in msg for word in ["sell", "share", "third", "advertising", "track", "never", "practice"]):
                    return f"HIDDEN_PRACTICE: {issue.get('severity')}"
            
            # --- Hard task: contradiction / dark pattern / false statement ---
            elif issue_type in ["contradiction", "dark_pattern"]:
                if "never" in msg or "not" in msg or "contradict" in msg or "section" in msg:
                    return f"CONTRADICTION: {issue.get('severity')}"
            
            elif issue_type == "false_statement":
                if any(word in msg for word in ["never", "not", "contradict", "false", "minimal", "necessary", "tracking"]):
                    return f"FALSE_STATEMENT: {issue.get('severity')}"
            
            # --- Shared: deceptive rights ---
            elif issue_type == "deceptive_rights":
                if "deletion" in msg or "delete" in msg or "right" in msg or "continue" in msg:
                    return f"DECEPTIVE_CLAUSE: {issue.get('severity')}"
            
            # --- Shared: missing disclosure ---
            elif issue_type == "missing_disclosure":
                if "third" in msg or "partner" in msg or "disclose" in msg or "shared" in msg:
                    return f"MISSING_DISCLOSURE: {issue.get('severity')}"
        
        # Fallback: generic finding if agent mentions relevant terms
        if any(word in msg for word in ["issue", "violation", "problem", "concern", "missing", "contradict"]):
            return "GENERAL_FINDING"
        
        return None
    
    def _calculate_reward(self) -> RewModel:
        # Scores must be strictly in (0, 1) — never exactly 0.0 or 1.0
        _EPSILON = 1e-6
        _MIN_SCORE = _EPSILON
        _MAX_SCORE = 1.0 - _EPSILON

        task = self._ep.task_config
        total_issues = len(task.hidden_issues)
        found_count = len(self._ep.found_issues)
        
        base_score = found_count / total_issues if total_issues > 0 else 0.0
        
        severity_bonus = 0.0
        critical_found = any("critical" in f.lower() for f in self._ep.found_issues)
        high_found = any("high" in f.lower() for f in self._ep.found_issues)
        
        if critical_found:
            severity_bonus += 0.25
        if high_found:
            severity_bonus += 0.15
        
        multi_doc_bonus = 0.0
        if task.difficulty == "elite":
            multi_doc_found = any("multi_doc" in f.lower() for f in self._ep.found_issues)
            if multi_doc_found:
                multi_doc_bonus += 0.2
        
        exploration_bonus = min(self._ep.steps * 0.02, 0.1)
        
        raw_reward = base_score + severity_bonus + multi_doc_bonus + exploration_bonus
        # Clamp to strictly open interval (0, 1)
        total_reward = max(_MIN_SCORE, min(_MAX_SCORE, raw_reward))
        
        reason = f"Found {found_count}/{total_issues} issues"
        
        return RewModel(
            value=total_reward,
            reason=reason,
            issues_found=found_count,
            total_issues=total_issues,
        )
    
    def _build_observation(self, message: str) -> ObsModel:
        if self._ep is None:
            return self._error_obs()
            
        return ObsModel(
            task_id=self._ep.task_config.task_id,
            task_name=self._ep.task_config.name,
            difficulty=self._ep.task_config.difficulty,
            step=self._ep.steps,
            documents=self._ep.documents,
            data_practices=self._ep.data_practices,
            compliance_requirements=self._ep.task_config.compliance_requirements,
            flagged_issues=self._ep.found_issues,
            echoed_message=message,
        )
    
    def _error_obs(self, message: str = "Error: Environment not initialized") -> ObsModel:
        return ObsModel(
            task_id="",
            task_name="",
            difficulty="",
            step=0,
            documents=[],
            data_practices=[],
            compliance_requirements=[],
            flagged_issues=[],
            echoed_message=message,
        )
    
    def state(self) -> Dict[str, Any]:
        if self._ep is None:
            return {}
        
        return {
            "episode_id": self._ep.episode_id,
            "task_id": self._ep.task_config.task_id,
            "task_name": self._ep.task_config.name,
            "difficulty": self._ep.task_config.difficulty,
            "steps": self._ep.steps,
            "found_issues": self._ep.found_issues,
            "total_issues": len(self._ep.task_config.hidden_issues),
        }


Environment = GDPRAuditorEnvironment
