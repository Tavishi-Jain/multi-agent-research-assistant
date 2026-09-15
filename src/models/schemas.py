"""
Pydantic schemas for the multi-agent research system.

Defines structured data models for:
- ReportSection: individual sections of the final report
- Source: metadata about a web source
- ResearchReport: the full drafted/final report
- Critique: structured feedback from the critic agent
- ResearchState: the complete state tracked through the LangGraph loop
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single web source used in the report."""
    
    url: str = Field(..., description="Full URL of the source")
    title: str = Field(..., description="Title or headline from the source")
    snippet: str = Field(..., description="Brief excerpt or summary from the source")
    retrieved_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this source was retrieved"
    )


class ReportSection(BaseModel):
    """A single section of the research report."""
    
    heading: str = Field(..., description="Section heading/title")
    content: str = Field(..., description="Body text of the section")
    source_indices: list[int] = Field(
        default_factory=list,
        description="Indices into ResearchReport.sources_used for citations"
    )


class ResearchReport(BaseModel):
    """The complete research report draft."""
    
    title: str = Field(..., description="Report title based on the topic")
    sections: list[ReportSection] = Field(
        default_factory=list,
        description="Ordered sections of the report"
    )
    sources_used: list[Source] = Field(
        default_factory=list,
        description="All sources cited in the report"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional executive summary or abstract"
    )


class CritiquePoint(BaseModel):
    """A single critique point from the critic agent."""
    
    category: str = Field(
        ...,
        description="Category: e.g. 'missing_info', 'unsupported_claim', 'clarity', 'structure'"
    )
    severity: str = Field(
        ...,
        description="Severity level: 'low', 'medium', 'high'"
    )
    description: str = Field(..., description="Detailed description of the issue")
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Suggested how to fix it"
    )


class Critique(BaseModel):
    """Structured critique output from the critic agent."""
    
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Quality score out of 10; >= threshold means pass"
    )
    is_passing: bool = Field(
        ...,
        description="True if quality_score >= threshold, else False"
    )
    critique_points: list[CritiquePoint] = Field(
        default_factory=list,
        description="List of specific issues found"
    )
    overall_feedback: str = Field(
        ...,
        description="Summary feedback from the critic"
    )
    groundedness_check: bool = Field(
        default=True,
        description="True if all major claims traced to sources, else False"
    )


class ResearchState(BaseModel):
    """
    Complete state tracked through the LangGraph loop.
    
    Passed through each node and updated as the researcher and critic
    iterate on the draft report.
    """
    
    topic: str = Field(..., description="Original research topic/query")
    current_draft: Optional[ResearchReport] = Field(
        default=None,
        description="Current draft of the report (None initially)"
    )
    critique_history: list[Critique] = Field(
        default_factory=list,
        description="All critiques from each iteration"
    )
    iteration_count: int = Field(
        default=0,
        description="How many research-critique cycles completed"
    )
    max_iterations: int = Field(
        default=3,
        description="Maximum iterations before forced termination"
    )
    quality_threshold: float = Field(
        default=8.0,
        ge=0.0,
        le=10.0,
        description="Quality score needed to pass and exit loop"
    )
    search_errors: list[str] = Field(
        default_factory=list,
        description="Log of any web search failures encountered"
    )
    is_complete: bool = Field(
        default=False,
        description="True if loop exited (either pass or max_iterations)"
    )
    final_report: Optional[ResearchReport] = Field(
        default=None,
        description="Final report when is_complete=True"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
