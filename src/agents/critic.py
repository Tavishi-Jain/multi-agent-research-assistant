"""
Critic agent: reviews research reports and provides structured feedback.

The critic evaluates a report against the original topic, checking for:
- Groundedness: all claims traced to sources
- Completeness: major gaps in coverage
- Clarity: is the content understandable?
- Accuracy: are claims reasonable given sources?

Returns a structured Critique with a quality_score (0-10) and detailed feedback.
"""

import logging
from typing import Optional
import json

from anthropic import Anthropic

from src.models.schemas import ResearchReport, Critique, CritiquePoint, ResearchState

logger = logging.getLogger(__name__)


class CriticAgent:
    """Agent responsible for critiquing research reports."""
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        quality_threshold: float = 8.0,
    ):
        """
        Initialize the critic agent.
        
        Args:
            model: Anthropic model to use
            quality_threshold: Quality score above which report passes
        """
        self.model = model
        self.quality_threshold = quality_threshold
        self.client = Anthropic()
    
    def _build_groundedness_check(self, report: ResearchReport) -> tuple[bool, list[str]]:
        """
        Check if all major claims in the report are traceable to sources.
        
        This is a lightweight heuristic check: ensures sections reference sources.
        
        Args:
            report: ResearchReport to check
            
        Returns:
            Tuple of (is_grounded: bool, issues: list[str])
        """
        issues = []
        
        # Check each section has at least one source reference
        for section in report.sections:
            if not section.source_indices:
                issues.append(
                    f"Section '{section.heading}' has no source citations"
                )
        
        # Check that all referenced source indices are valid
        for section in report.sections:
            for idx in section.source_indices:
                if idx < 0 or idx >= len(report.sources_used):
                    issues.append(
                        f"Section '{section.heading}' references invalid source index {idx}"
                    )
        
        is_grounded = len(issues) == 0
        
        if not is_grounded:
            logger.warning(f"Groundedness issues found: {issues}")
        else:
            logger.info("Groundedness check passed")
        
        return is_grounded, issues
    
    def _get_critique_from_claude(
        self,
        topic: str,
        report: ResearchReport,
    ) -> dict:
        """
        Use Claude to evaluate the report and return structured critique.
        
        Args:
            topic: Original research topic
            report: ResearchReport to critique
            
        Returns:
            Dictionary with quality_score, critique_points, and overall_feedback
        """
        sources_summary = "\n".join([
            f"- {s.title} ({s.url})"
            for s in report.sources_used
        ])
        
        sections_summary = "\n".join([
            f"## {s.heading}\n{s.content[:300]}..."
            for s in report.sections
        ])
        
        prompt = f"""
You are an expert research critic. Evaluate the following research report against the original topic.

**Original Topic:** {topic}

**Report Title:** {report.title}

**Report Sections:**
{sections_summary}

**Sources Used:**
{sources_summary if sources_summary else "No sources cited"}

**Evaluation Rubric:**
1. **Completeness**: Does it cover major aspects of the topic? Are there obvious gaps?
2. **Accuracy**: Are the claims reasonable and supported by the sources?
3. **Clarity**: Is the writing clear and well-organized?
4. **Groundedness**: Are claims properly cited to sources?
5. **Depth**: Does it provide sufficient detail and insight?

Provide your critique in the following JSON format:
{{
    "quality_score": <integer 0-10>,
    "critique_points": [
        {{
            "category": "<one of: missing_info, unsupported_claim, clarity, structure, accuracy, depth>",
            "severity": "<low|medium|high>",
            "description": "<specific issue>",
            "suggested_fix": "<how to address it>"
        }},
        ...
    ],
    "overall_feedback": "<1-2 sentence summary of the report's quality>"
}}

Be honest and constructive. A score of 8+ indicates a high-quality report ready to publish.
A score below 8 indicates room for improvement.
"""
        
        logger.info("Requesting critique from Claude")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        logger.debug(f"Claude critique response: {response_text[:500]}...")
        
        # Extract JSON from response
        if "```json" in response_text:
            start = response_text.index("```json") + 7
            end = response_text.index("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.index("```") + 3
            end = response_text.index("```", start)
            json_str = response_text[start:end].strip()
        else:
            json_str = response_text
        
        data = json.loads(json_str)
        
        return data
    
    def critique(
        self,
        state: ResearchState,
    ) -> ResearchState:
        """
        Main critic node: evaluate the current draft and update state.
        
        Args:
            state: Current ResearchState (must have current_draft set)
            
        Returns:
            Updated ResearchState with new Critique appended to critique_history
        """
        if state.current_draft is None:
            logger.error("No current draft to critique")
            raise ValueError("ResearchState.current_draft is None")
        
        logger.info(f"Critic evaluating iteration {state.iteration_count}")
        logger.info(f"Report title: {state.current_draft.title}")
        
        # Check groundedness first
        is_grounded, groundedness_issues = self._build_groundedness_check(state.current_draft)
        
        # Get structured critique from Claude
        critique_data = self._get_critique_from_claude(
            state.topic,
            state.current_draft,
        )
        
        # Parse quality score
        quality_score = float(critique_data.get("quality_score", 0))
        
        # Clamp to 0-10
        quality_score = max(0.0, min(10.0, quality_score))
        
        # Parse critique points
        critique_points = []
        for point_data in critique_data.get("critique_points", []):
            point = CritiquePoint(
                category=point_data.get("category", "other"),
                severity=point_data.get("severity", "medium"),
                description=point_data.get("description", ""),
                suggested_fix=point_data.get("suggested_fix"),
            )
            critique_points.append(point)
        
        # If groundedness check failed, add it as a critique point
        if not is_grounded:
            for issue in groundedness_issues:
                point = CritiquePoint(
                    category="groundedness",
                    severity="high",
                    description=issue,
                    suggested_fix="Add source citations or remove unsupported claims",
                )
                critique_points.append(point)
            
            # Reduce quality score slightly if groundedness issues
            quality_score = min(quality_score, 7.0)
        
        # Create Critique object
        is_passing = quality_score >= state.quality_threshold
        
        critique = Critique(
            quality_score=quality_score,
            is_passing=is_passing,
            critique_points=critique_points,
            overall_feedback=critique_data.get("overall_feedback", ""),
            groundedness_check=is_grounded,
        )
        
        # Update state
        state.critique_history.append(critique)
        
        logger.info(f"Critique complete: score={quality_score}, passing={is_passing}")
        logger.info(f"Issues found: {len(critique_points)}")
        
        return state
