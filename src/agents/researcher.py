"""
Researcher agent: drafts and revises research reports.

The researcher accepts a topic and optional prior critique, searches the web
for relevant information, and produces a structured ResearchReport. On revision
iterations, it incorporates feedback from the critic.
"""

import logging
from typing import Optional

from anthropic import Anthropic

from src.models.schemas import ResearchReport, ReportSection, Source, ResearchState
from src.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """Agent responsible for drafting and revising research reports."""
    
    def __init__(
        self,
        search_tool: WebSearchTool,
        model: str = "claude-3-5-sonnet-20241022",
        max_search_queries: int = 5,
        max_section_depth: int = 5,
    ):
        """
        Initialize the researcher agent.
        
        Args:
            search_tool: WebSearchTool instance for searching
            model: Anthropic model to use
            max_search_queries: Maximum number of web searches per draft
            max_section_depth: Maximum number of sections in a report
        """
        self.search_tool = search_tool
        self.model = model
        self.max_search_queries = max_search_queries
        self.max_section_depth = max_section_depth
        self.client = Anthropic()
    
    def _build_search_queries(self, topic: str, prior_critique: Optional[str] = None) -> list[str]:
        """
        Generate search queries for the topic.
        
        If there's prior critique, focus on the gaps/issues identified.
        
        Args:
            topic: Research topic
            prior_critique: Optional feedback from previous iteration
            
        Returns:
            List of search queries
        """
        prompt = f"""
Given the research topic: "{topic}"

Generate {self.max_search_queries} diverse web search queries to gather comprehensive information.
{"Additional context - focus on addressing: " + prior_critique if prior_critique else ""}

Return queries as a simple numbered list, one per line, without additional commentary.
Example format:
1. query one
2. query two
...
"""
        
        logger.info(f"Generating search queries for topic: {topic}")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        queries = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering and bullet points
                query = line.lstrip("0123456789.-) ").strip()
                if query:
                    queries.append(query)
        
        logger.info(f"Generated {len(queries)} search queries")
        return queries[:self.max_search_queries]
    
    def _search_and_aggregate(self, queries: list[str]) -> tuple[list[Source], dict]:
        """
        Execute searches and aggregate results.
        
        Args:
            queries: List of search queries
            
        Returns:
            Tuple of (list of deduplicated Source objects, dict of errors encountered)
        """
        all_sources = []
        seen_urls = set()
        search_errors = {}
        
        for query in queries:
            try:
                logger.info(f"Executing search: {query}")
                results = self.search_tool.search(query, max_results=3)
                
                for source in results:
                    if source.url not in seen_urls:
                        all_sources.append(source)
                        seen_urls.add(source.url)
                
                logger.info(f"Search '{query}' returned {len(results)} results")
            
            except Exception as e:
                error_msg = f"Search failed for '{query}': {str(e)}"
                logger.error(error_msg)
                search_errors[query] = str(e)
        
        logger.info(f"Total unique sources after aggregation: {len(all_sources)}")
        return all_sources, search_errors
    
    def _draft_report(
        self,
        topic: str,
        sources: list[Source],
        prior_critique: Optional[str] = None,
    ) -> ResearchReport:
        """
        Use Claude to draft a structured report from sources.
        
        Args:
            topic: Research topic
            sources: List of Source objects retrieved
            prior_critique: Optional feedback to address
            
        Returns:
            ResearchReport object
        """
        sources_text = "\n\n".join([
            f"[Source {i+1}] Title: {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
            for i, s in enumerate(sources)
        ])
        
        prompt = f"""
Research Topic: {topic}

Available Sources:
{sources_text if sources_text else "No sources available - provide general knowledge-based response."}

{f"Prior Feedback to Address: {prior_critique}" if prior_critique else ""}

Draft a comprehensive research report on the topic. Structure it with:
1. A clear, descriptive title
2. 3-5 main sections, each with a heading and substantive content
3. Each section should cite sources by number (e.g., [Source 1]) where applicable

Format your response as JSON with this exact structure:
{{
    "title": "Report Title",
    "summary": "Brief 1-2 sentence executive summary",
    "sections": [
        {{"heading": "Section Name", "content": "Section content with [Source N] citations"}},
        ...
    ]
}}

Ensure all claims are supported by the provided sources. Be thorough but concise.
"""
        
        logger.info("Drafting report using Claude")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        logger.debug(f"Claude response: {response_text[:500]}...")
        
        # Parse JSON response
        import json
        
        # Extract JSON from response (handle markdown code blocks)
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
        
        # Build ReportSection objects with source indices
        sections = []
        for section_data in data.get("sections", []):
            # Extract source references from content
            content = section_data.get("content", "")
            source_indices = []
            
            # Parse [Source N] references
            import re
            for match in re.finditer(r"\[Source (\d+)\]", content):
                idx = int(match.group(1)) - 1  # Convert to 0-indexed
                if 0 <= idx < len(sources):
                    source_indices.append(idx)
            
            # Remove duplicates and sort
            source_indices = sorted(list(set(source_indices)))
            
            section = ReportSection(
                heading=section_data.get("heading", ""),
                content=content,
                source_indices=source_indices,
            )
            sections.append(section)
        
        report = ResearchReport(
            title=data.get("title", topic),
            summary=data.get("summary", ""),
            sections=sections,
            sources_used=sources,
        )
        
        logger.info(f"Drafted report with {len(sections)} sections and {len(sources)} sources")
        return report
    
    def research(
        self,
        state: ResearchState,
    ) -> ResearchState:
        """
        Main researcher node: search, draft/revise, and update state.
        
        Args:
            state: Current ResearchState
            
        Returns:
            Updated ResearchState with new draft and search errors logged
        """
        logger.info(f"Researcher starting iteration {state.iteration_count + 1}")
        logger.info(f"Topic: {state.topic}")
        
        # Build critique summary if we have prior feedback
        prior_critique_text = None
        if state.critique_history:
            last_critique = state.critique_history[-1]
            prior_critique_text = (
                f"Quality score: {last_critique.quality_score}/10. "
                f"Issues: {'; '.join([f'{p.category}: {p.description}' for p in last_critique.critique_points])}"
            )
            logger.info(f"Incorporating prior critique: {prior_critique_text}")
        
        # Generate search queries
        queries = self._build_search_queries(
            state.topic,
            prior_critique=prior_critique_text
        )
        
        # Execute searches
        sources, search_errors = self._search_and_aggregate(queries)
        
        # Log search errors to state
        for query, error in search_errors.items():
            error_log = f"Query '{query}': {error}"
            state.search_errors.append(error_log)
            logger.warning(error_log)
        
        # Draft report
        report = self._draft_report(
            state.topic,
            sources,
            prior_critique=prior_critique_text
        )
        
        # Update state
        state.current_draft = report
        state.iteration_count += 1
        
        logger.info(f"Researcher completed iteration {state.iteration_count}")
        return state
