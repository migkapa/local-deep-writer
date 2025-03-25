import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from ollama_deep_researcher.configuration import Configuration, SearchAPI
from ollama_deep_researcher.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, duckduckgo_search, searxng_search, strip_thinking_tokens, get_config_value
from ollama_deep_researcher.state import SummaryState, SummaryStateInput, SummaryStateOutput
from ollama_deep_researcher.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions, get_current_date
from ollama_deep_researcher.lmstudio import ChatLMStudio

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """LangGraph node that generates a search query based on the research topic.
    
    Uses an LLM to create an optimized search query for web research based on
    the user's research topic. Supports both LMStudio and Ollama as LLM providers.
    
    Args:
        state: Current graph state containing the research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state.research_topic
    )

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=formatted_prompt),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    
    # Get the content
    content = result.content

    # Parse the JSON response and get the query
    try:
        query = json.loads(content)
        search_query = query['query']
    except (json.JSONDecodeError, KeyError):
        # If parsing fails or the key is not found, use a fallback query
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = content
    return {"search_query": search_query}

def web_research(state: SummaryState, config: RunnableConfig):
    """LangGraph node that performs web research using the generated search query.
    
    Executes a web search using the configured search API (tavily, perplexity, 
    duckduckgo, or searxng) and formats the results for further processing.
    
    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings
        
    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Get the search API
    search_api = get_config_value(configurable.search_api)

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, fetch_full_page=configurable.fetch_full_page, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "duckduckgo":
        search_results = duckduckgo_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    elif search_api == "searxng":
        search_results = searxng_search(state.search_query, max_results=3, fetch_full_page=configurable.fetch_full_page)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, fetch_full_page=configurable.fetch_full_page)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def article_writer(state: SummaryState, config: RunnableConfig):
    """LangGraph node that creates or enhances an SEO-optimized article.
    
    Uses an LLM to create or update an article based on the newest web research 
    results, integrating them with any existing content and optimizing for SEO.
    
    Args:
        state: Current graph state containing research topic, article content,
              web research results, and SEO keywords
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including article_content and seo_keywords
    """

    # Existing article content
    existing_article = state.article_content

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]
    
    # Get SEO keywords - either from state or from generate_query result
    seo_keywords = state.seo_keywords if state.seo_keywords else []
    
    # Format SEO keywords for the prompt
    seo_keywords_str = ", ".join(seo_keywords) if seo_keywords else "No specific keywords identified yet"

    # Build the human message
    if existing_article:
        human_message_content = (
            f"WRITE STRICTLY ABOUT THIS TOPIC: '{state.research_topic}'\n\n"
            f"<Article Topic> \n {state.research_topic} \n </Article Topic>\n\n"
            f"<SEO Keywords> \n {seo_keywords_str} \n </SEO Keywords>\n\n"
            f"<Existing Article> \n {existing_article} \n </Existing Article>\n\n"
            f"<New Research> \n {most_recent_web_research} \n </New Research>\n\n"
            f"ARTICLE STRUCTURE REQUIREMENTS:\n"
            f"1. Begin with a 'In This Article' table of contents with anchor links to each section\n"
            f"2. Write a substantial, in-depth article (1500-2000+ words)\n"
            f"3. Organize into 5-7 well-structured sections with descriptive headers\n"
            f"4. Each section should have at least 3-4 paragraphs of detailed content\n"
            f"5. Use proper HTML anchors for all section headers (<h2 id=\"section-id\">Section Title</h2>)\n"
            f"6. Include comparison tables, lists, and examples where relevant\n\n"
            f"IMPORTANT: Your entire article must focus ONLY on '{state.research_topic}' using ONLY the provided research materials."
        )
    else:
        human_message_content = (
            f"WRITE STRICTLY ABOUT THIS TOPIC: '{state.research_topic}'\n\n"
            f"<Article Topic> \n {state.research_topic} \n </Article Topic>\n\n"
            f"<SEO Keywords> \n {seo_keywords_str} \n </SEO Keywords>\n\n"
            f"<Research Materials> \n {most_recent_web_research} \n </Research Materials>\n\n"
            f"ARTICLE STRUCTURE REQUIREMENTS:\n"
            f"1. Begin with a 'In This Article' table of contents with anchor links to each section\n"
            f"2. Write a substantial, in-depth article (1500-2000+ words)\n"
            f"3. Organize into 5-7 well-structured sections with descriptive headers\n"
            f"4. Each section should have at least 3-4 paragraphs of detailed content\n"
            f"5. Use proper HTML anchors for all section headers (<h2 id=\"section-id\">Section Title</h2>)\n"
            f"6. Include comparison tables, lists, and examples where relevant\n\n"
            f"IMPORTANT: Your entire article must focus ONLY on '{state.research_topic}' using ONLY the provided research materials."
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0
        )
    else:  # Default to Ollama
        llm = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0
        )
    
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    # Strip thinking tokens if configured
    article_content = result.content
    if configurable.strip_thinking_tokens:
        article_content = strip_thinking_tokens(article_content)

    return {"article_content": article_content}

def seo_analysis(state: SummaryState, config: RunnableConfig):
    """LangGraph node that analyzes article content for SEO optimization opportunities.
    
    Analyzes the current article to identify content gaps, SEO opportunities, and generates
    follow-up queries to enhance article quality and search ranking. Uses structured output 
    to extract SEO recommendations and follow-up queries in JSON format.
    
    Args:
        state: Current graph state containing the article content and research topic
        config: Configuration for the runnable, including LLM provider settings
        
    Returns:
        Dictionary with state update, including search_query and seo_keywords
    """

    # Get previous queries to avoid repetition
    previous_queries = []
    if hasattr(state, 'search_query') and state.search_query:
        previous_queries.append(state.search_query)
    
    # Format previous queries
    prev_queries_str = ", ".join(previous_queries) if previous_queries else "No previous queries"

    # Generate SEO analysis
    configurable = Configuration.from_runnable_config(config)
    
    # Choose the appropriate LLM based on the provider
    if configurable.llm_provider == "lmstudio":
        llm_json_mode = ChatLMStudio(
            base_url=configurable.lmstudio_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    else: # Default to Ollama
        llm_json_mode = ChatOllama(
            base_url=configurable.ollama_base_url, 
            model=configurable.local_llm, 
            temperature=0, 
            format="json"
        )
    
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(
            research_topic=state.research_topic,
            previous_queries=prev_queries_str,
            current_summary=state.article_content or "No article content yet."
        )),
        HumanMessage(content=f"Analyze this article for SEO optimization and content improvement opportunities:")]
    )
    
    # Process the result
    try:
        # Try to parse as JSON first
        seo_analysis = json.loads(result.content)
        
        # Get the follow-up query
        query = seo_analysis.get('follow_up_query')
        
        # Get SEO keywords and ensure they're unique
        new_keywords = seo_analysis.get('target_keywords', [])
        
        # Extract section gaps and content opportunities if available
        section_gaps = seo_analysis.get('section_gaps', [])
        content_opportunities = seo_analysis.get('content_opportunities', [])
        
        # For logging/debugging - print what we found
        print(f"Found {len(new_keywords)} keywords, {len(section_gaps)} section gaps, and {len(content_opportunities)} content opportunities")
        
        # Check if query is None or empty
        if not query:
            # Use a fallback query
            query = f"Tell me more about {state.research_topic}"
            
        # Add the new keywords to the state - properly deduplicate
        existing_keywords = state.seo_keywords if state.seo_keywords else []
        
        # Normalize and deduplicate keywords (case-insensitive)
        normalized_existing = [k.lower().strip() for k in existing_keywords]
        all_keywords = existing_keywords.copy()
        
        for keyword in new_keywords:
            # Only add if normalized version isn't already in list
            normalized = keyword.lower().strip()
            if normalized not in normalized_existing:
                all_keywords.append(keyword)
                normalized_existing.append(normalized)
            
        return {"search_query": query, "seo_keywords": all_keywords}
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If parsing fails or the key is not found, use a fallback query
        return {"search_query": f"Tell me more about {state.research_topic}"}
        
def finalize_article(state: SummaryState):
    """LangGraph node that finalizes the SEO-optimized article.
    
    Prepares the final output with proper HTML formatting including headlines, paragraphs,
    and table of contents with anchor links. Returns only the clean, formatted article
    without any sources or additional metadata.
    
    Args:
        state: Current graph state containing the article content and SEO keywords
        
    Returns:
        Dictionary with state update, including article_content with HTML formatting and SEO keywords
    """

    # The article content should already be properly formatted from the article_writer node
    # with HTML anchors and structure from our instructions in the prompt
    
    # Format the article with the topic as title, but keep all the HTML formatting
    # that was generated in the article_writer step
    clean_article = f"""<h1>{state.research_topic}</h1>

{state.article_content}"""
    
    # Return just the clean article with HTML formatting and SEO keywords
    return {"article_content": clean_article, "seo_keywords": state.seo_keywords}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_article", "web_research"]:
    """LangGraph routing function that determines the next step in the content creation flow.
    
    Controls the research loop by deciding whether to continue gathering information
    or to finalize the article based on the configured maximum number of research loops.
    
    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_web_research_loops setting
        
    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_article")
    """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_article"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("article_writer", article_writer)
builder.add_node("seo_analysis", seo_analysis)
builder.add_node("finalize_article", finalize_article)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "article_writer")
builder.add_edge("article_writer", "seo_analysis")
builder.add_conditional_edges("seo_analysis", route_research)
builder.add_edge("finalize_article", END)

graph = builder.compile()