import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated

@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Article topic     
    search_query: str = field(default=None) # Search query
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list) 
    research_loop_count: int = field(default=0) # Research loop count
    article_content: str = field(default=None) # The evolving article content
    html_content: str = field(default=None) # The HTML version of the article content
    seo_keywords: Annotated[list, operator.add] = field(default_factory=list) # SEO keywords identified

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None) # Article topic     

@dataclass(kw_only=True)
class SummaryStateOutput:
    article_content: str = field(default=None) # Final article in markdown
    html_content: str = field(default=None) # Final article in HTML
    seo_keywords: list = field(default_factory=list) # SEO keywords used in the article