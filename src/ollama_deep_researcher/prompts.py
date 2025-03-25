from datetime import datetime

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")

query_writer_instructions="""Your goal is to generate a targeted web search query to gather information for an SEO-optimized article.

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current information available as of this date.
You're gathering information to write a high-quality, SEO-optimized article that will rank well in search results.
</CONTEXT>

<TOPIC>
{research_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with these exact keys:
   - "query": The actual search query string
   - "seo_keywords": List of 3-5 important SEO keywords related to this topic
   - "rationale": Brief explanation of why this query will help gather information for a high-quality article
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "latest machine learning transformer architecture advancements 2025",
    "seo_keywords": ["transformer architecture", "machine learning advancements", "AI models 2025", "neural network developments"],
    "rationale": "This query will gather the most recent technical developments in transformer models to create a comprehensive and current article that attracts readers interested in cutting-edge AI technology."
}}
</EXAMPLE>

Provide your response in JSON format:"""

summarizer_instructions="""
<GOAL>
Write a comprehensive, in-depth, and lengthy article based EXCLUSIVELY on the provided web search results that directly addresses the article topic. The article should be SEO-optimized, well-structured, and include a table of contents with anchor links. You must NOT include any information that is not present in the provided research materials.
</GOAL>

<REQUIREMENTS>
IMPORTANT: 
- ONLY write about the EXACT article topic provided - do not change the subject or add unrelated content
- NEVER generate content about AI, healthcare, or other topics unless they are directly mentioned in the research materials
- ONLY use information from the provided research materials - do not add information from your training data
- Create a SUBSTANTIAL article with at least 1500-2000 words of in-depth content

When creating a NEW article:
1. Begin with a table of contents titled "In This Article" with anchor links to each section (use HTML anchor format)
2. Write a compelling introduction that hooks the reader and previews the main sections
3. Organize content into at least 5-7 clearly defined sections with descriptive headings
4. Include subsections where appropriate for detailed coverage of specific aspects
5. Ensure each section thoroughly explores its topic with relevant details, examples, and insights
6. Maintain an informative, authoritative tone throughout
7. Include comparison tables, pros/cons lists, or rating systems where relevant
8. End with a comprehensive conclusion that summarizes key points and provides final recommendations

When EXTENDING an existing article:                                                                                                                 
1. Carefully review both the existing article and new search results                                                    
2. Identify gaps, inconsistencies, or areas that need expansion                                                         
3. For each piece of new information:                                                                             
    a. Seamlessly integrate related content into appropriate sections                               
    b. Add new sections for significant new information with smooth transitions                            
    c. Filter out irrelevant information that doesn't serve the article topic                                                            
4. Update the table of contents to reflect any new sections
5. Expand existing sections with more detailed information
6. Ensure all additions strengthen the article's focus and depth                                                         
7. Produce a substantially improved and expanded version of the original article without deviating from the topic                                                                                                                                                            
</REQUIREMENTS>

<FORMATTING>
- Begin with a table of contents using this format:
  ## In This Article
  - [Section 1 Title](#section-1-title)
  - [Section 2 Title](#section-2-title)
  ...

- Use clear hierarchical structure with H2 for main sections and H3 for subsections
- Include descriptive headers that incorporate key SEO terms
- Use bullet points, numbered lists, and comparison tables where appropriate
- For each section, use the format: <h2 id="section-id">Section Title</h2>
- Make each section substantive - at least 3-4 paragraphs of detailed content
</FORMATTING>

<IMPORTANT WARNING>
You MUST ensure the ENTIRE article focuses ONLY on the EXACT article topic provided. Check carefully before submission.
</IMPORTANT WARNING>
"""

reflection_instructions = """You are an expert content strategist and SEO specialist analyzing a structured, in-depth article about {research_topic}.

<CONTEXT>
Current article content: {current_summary}
Previous search queries: {previous_queries}
</CONTEXT>

<GOAL>
1. Perform a comprehensive analysis of this article to identify content gaps, sections that need deeper exploration, and opportunities for SEO improvement
2. Generate a strategic follow-up query that would help enhance the article's quality, depth, and search ranking potential
3. Identify sections that could be expanded or new sections that should be added
4. Focus on high-value content, trending topics, and SEO best practices that weren't fully covered
</GOAL>

<REQUIREMENTS>
1. Consider which sections of the article need more depth and specific details
2. Identify specific content types that could enhance the article (case studies, statistics, expert opinions, comparisons)
3. Analyze keyword density and opportunities for better keyword integration
4. Provide specific, actionable guidance rather than general suggestions
5. Ensure the follow-up query is search-engine optimized and includes necessary context for effective web search
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- section_gaps: Identify 2-3 specific sections that need expansion or new sections that should be added
- content_opportunities: List 3-4 specific content types or examples that would enhance the article
- seo_opportunities: List 2-3 specific ways the article could be improved for search rankings
- follow_up_query: Write a specific question to address the identified gaps
- target_keywords: List 5-7 important keywords that should be incorporated (no duplicates)
</FORMAT>

<EXAMPLE>
Example output:
{{
    "section_gaps": ["The article needs a dedicated section on real-world implementation case studies", "Adding a troubleshooting section would address common user pain points", "A comparison section between different solutions would help readers make decisions"],
    "content_opportunities": ["Include expert opinions from industry leaders", "Add statistical data on user adoption rates", "Create a step-by-step tutorial with screenshots", "Include a comparison table of features across different options"],
    "seo_opportunities": ["Add more H2 and H3 headings with target keywords", "Include more statistical data and case studies", "Add comparison sections with competing products"],
    "follow_up_query": "What are the most successful case studies and implementation strategies for [specific product] in home environments in 2025?",
    "target_keywords": ["best practices", "comparison guide", "product reviews", "troubleshooting guide", "expert recommendations", "buyer's guide 2025", "product installation"]
}}
</EXAMPLE>

Provide your analysis in JSON format:"""