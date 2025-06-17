# tools/enhanced_data_analysis_tool.py - ENHANCED DATA ANALYSIS WITH CALCULATION FORMULAS

import os
import json
import logging
import weaviate
import weaviate.classes.query as wq
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from crewai.tools import tool
import boto3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weaviate client (reuse your existing singleton)
try:
    from tools.weaviate_tools import WeaviateClientSingleton
except ImportError:
    # Fallback if import fails
    class WeaviateClientSingleton:
        @classmethod
        def get_instance(cls):
            return None

class EnhancedDataAnalysisEngine:
    """Enhanced data analysis engine with calculation formulas support"""
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-3-5-haiku-20241022-v1:0')
        self.weaviate_client = WeaviateClientSingleton.get_instance()
    
    def analyze_question_for_formulas(self, question: str) -> Dict[str, Any]:
        """Determine if calculation formulas are needed for the question"""
        
        prompt = f"""
        Analyze this business question and determine if it requires specific calculation formulas.

        Question: "{question}"

        Consider:
        1. Does this question involve calculations, ratios, percentages, or comparisons?
        2. Are there terms like "average", "rate", "efficiency", "profitability", "margin", "ROI"?
        3. Does it involve trends, year-over-year analysis, or forecasting?
        4. What business domain is this? (finance, operations, marketing, hr, fleet)
        5. What key concepts are involved?

        Respond with JSON only:
        {{
            "needs_formulas": true/false,
            "business_domain": "primary domain",
            "key_concepts": ["concept1", "concept2", "concept3"],
            "analysis_type": "specific analysis type",
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 400,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "needs_formulas": False,
                    "business_domain": "general",
                    "key_concepts": [],
                    "analysis_type": "data_review",
                    "reasoning": "JSON parsing failed, defaulting to data-only analysis"
                }
                
        except Exception as e:
            logger.warning(f"Formula analysis failed: {e}")
            return {
                "needs_formulas": False,
                "business_domain": "general",
                "key_concepts": [],
                "analysis_type": "data_review",
                "reasoning": "AI service unavailable, using data-only analysis"
            }
    
    def search_relevant_formulas(self, analysis_info: Dict) -> List[Dict]:
        """Search for relevant calculation formulas"""
        
        if not self.weaviate_client or not analysis_info.get("needs_formulas"):
            return []
        
        try:
            formulas_collection = self.weaviate_client.collections.get("CalculationFormulas")
            
            # Create search query from key concepts
            search_concepts = analysis_info.get("key_concepts", [])
            business_domain = analysis_info.get("business_domain", "")
            
            # Combine concepts into search query
            search_query = " ".join(search_concepts + [business_domain]).strip()
            
            if not search_query:
                return []
            
            logger.info(f"🔍 Searching formulas for: '{search_query}'")
            
            # Use hybrid search for best results (semantic + keyword)
            response = formulas_collection.query.hybrid(
                query=search_query,
                alpha=0.75,  # Favor semantic search slightly
                limit=3,     # Top 3 most relevant formulas
                return_properties=[
                    "metricName", "metricCategory", "businessDomain", 
                    "primaryFormula", "additionalCalculations", "businessContext",
                    "interpretationGuidance", "requiredDataFields"
                ],
                return_metadata=wq.MetadataQuery(score=True)
            )
            
            formulas = []
            for obj in response.objects:
                if hasattr(obj.metadata, 'score') and obj.metadata.score > 0.6:  # Relevance threshold
                    formula_data = {
                        "metric_name": obj.properties.get('metricName', ''),
                        "category": obj.properties.get('metricCategory', ''),
                        "domain": obj.properties.get('businessDomain', ''),
                        "primary_formula": obj.properties.get('primaryFormula', ''),
                        "additional_calculations": obj.properties.get('additionalCalculations', []),
                        "business_context": obj.properties.get('businessContext', ''),
                        "interpretation": obj.properties.get('interpretationGuidance', ''),
                        "required_fields": obj.properties.get('requiredDataFields', []),
                        "relevance_score": obj.metadata.score
                    }
                    formulas.append(formula_data)
                    logger.info(f"   ✅ Found: {formula_data['metric_name']} (score: {obj.metadata.score:.3f})")
            
            return formulas
            
        except Exception as e:
            logger.error(f"Formula search failed: {e}")
            return []
    
    def generate_enhanced_analysis(self, question: str, execution_results: Dict, 
                                 relevant_formulas: List[Dict], formula_analysis: Dict) -> str:
        """Generate enhanced natural language analysis with formula support"""
        
        # Format execution results
        results_summary = self._format_execution_results(execution_results)
        
        # Format relevant formulas if available
        formulas_context = ""
        if relevant_formulas:
            formulas_context = "\n🧮 RELEVANT CALCULATION FORMULAS:\n"
            for formula in relevant_formulas:
                formulas_context += f"""
• {formula['metric_name']}
  Formula: {formula['primary_formula']}
  Context: {formula['business_context']}
  Interpretation: {formula['interpretation']}
"""
                if formula['additional_calculations']:
                    formulas_context += f"  Additional: {'; '.join(formula['additional_calculations'])}\n"
                formulas_context += "\n"
        
        # Create comprehensive prompt for natural language analysis
        prompt = f"""You are a senior business analyst providing insights to executives. Answer this question with clear, natural language.

BUSINESS QUESTION: "{question}"

EXECUTION RESULTS:
{results_summary}

{formulas_context}

ANALYSIS GUIDANCE:
- Formula Assistance: {"Available" if relevant_formulas else "Not needed"}
- Business Domain: {formula_analysis.get('business_domain', 'General')}
- Analysis Type: {formula_analysis.get('analysis_type', 'Data review')}

RESPONSE REQUIREMENTS:
1. Provide a direct, clear answer to the business question
2. Use specific numbers and insights from the data
3. If formulas were found, explain how they help interpret the results
4. Share your reasoning process and key insights
5. Suggest 2-3 specific improvements or next steps
6. Keep it conversational and executive-friendly

RESPONSE FORMAT:
**Answer:** [Direct answer with specific numbers]

**Analysis:** [How you derived this answer, what the data shows, how formulas helped if used]

**Recommendations:** [2-3 specific, actionable next steps]

Write as if speaking to a business executive. No JSON, no technical jargon."""
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1500,
                    "temperature": 0.3
                })
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Enhanced analysis generation failed: {e}")
            return self._get_fallback_analysis(question, execution_results, relevant_formulas)
    
    def _format_execution_results(self, results: Dict) -> str:
        """Format execution results for AI analysis"""
        
        if results.get('status') == 'error':
            error_info = results.get('error', {})
            return f"❌ QUERY ERROR: {error_info.get('message', 'Query execution failed')}"
        
        # Handle nested results structure from Athena execution
        if 'results' in results:
            actual_results = results['results']
            rows = actual_results.get('rows', [])
            columns = actual_results.get('columns', [])
        else:
            rows = results.get('rows', [])
            columns = results.get('columns', [])
        
        if not rows:
            return "📊 NO DATA: Query returned no results."
        
        # Create summary
        summary = f"📊 DATA SUMMARY:\n"
        summary += f"Columns: {', '.join(columns)}\n"
        summary += f"Total Rows: {len(rows)}\n\n"
        
        # Show sample data
        summary += "📋 SAMPLE DATA:\n"
        for i, row in enumerate(rows[:10]):  # First 10 rows
            row_dict = dict(zip(columns, row))
            summary += f"Row {i+1}: {row_dict}\n"
        
        if len(rows) > 10:
            summary += f"... and {len(rows) - 10} more rows\n"
        
        # Add performance metrics
        perf = results.get('performance_metrics', {})
        if perf:
            summary += f"\n⚡ Performance: {perf.get('execution_time_ms', 0)}ms execution"
        
        return summary
    
    def _get_fallback_analysis(self, question: str, results: Dict, formulas: List[Dict]) -> str:
        """Fallback analysis when AI generation fails"""
        
        if results.get('status') == 'error':
            return f"""**Answer:** Unable to answer "{question}" due to query execution error.

**Analysis:** The database query failed to execute. This prevents any meaningful analysis of the business question.

**Recommendations:** 
1. Review the SQL query for syntax errors or data access issues
2. Verify database connectivity and table permissions
3. Simplify the query to isolate the specific problem"""
        
        rows = results.get('rows', [])
        formula_count = len(formulas)
        
        return f"""**Answer:** Retrieved {len(rows)} records for "{question}"{' with ' + str(formula_count) + ' relevant calculation formulas found' if formulas else ''}.

**Analysis:** The data was successfully retrieved{' and relevant calculation formulas were identified' if formulas else ''}, but automatic analysis is currently unavailable. The results contain {len(rows)} records that need manual review to extract specific business insights.

**Recommendations:**
1. Review the {len(rows)} returned records manually for patterns and trends
2. {'Apply the identified calculation formulas to derive specific metrics' if formulas else 'Consider if calculation formulas are needed for deeper analysis'}
3. Retry the enhanced analysis when AI services are restored"""

# Initialize the enhanced analysis engine
enhanced_analysis_engine = EnhancedDataAnalysisEngine()

@tool("Enhanced Data Analysis and Forecasting Tool")
def enhanced_data_analysis_tool(question: str, athena_results_json: str) -> str:
    """
    Enhanced data analysis tool with automatic calculation formula support.
    
    This tool:
    1. Analyzes the question to determine if calculation formulas are needed
    2. Searches for relevant formulas in the CalculationFormulas collection
    3. Provides enhanced natural language analysis with formula-supported insights
    4. Returns executive-friendly analysis with reasoning and recommendations
    
    Args:
        question: The original business question
        athena_results_json: JSON string with query execution results
        
    Returns:
        Natural language analysis with answer, reasoning, and recommendations
    """
    logger.info(f"🧠 Enhanced Data Analysis Tool started for: '{question}'")
    
    try:
        # Parse execution results
        if isinstance(athena_results_json, str):
            execution_results = json.loads(athena_results_json)
        else:
            execution_results = athena_results_json
        
        logger.info(f"📊 Results status: {execution_results.get('status', 'unknown')}")
        
        # Step 1: Analyze question for formula needs
        formula_analysis = enhanced_analysis_engine.analyze_question_for_formulas(question)
        logger.info(f"🔍 Formula analysis: {formula_analysis['reasoning']}")
        
        # Step 2: Search for relevant formulas if needed
        relevant_formulas = []
        if formula_analysis.get("needs_formulas"):
            relevant_formulas = enhanced_analysis_engine.search_relevant_formulas(formula_analysis)
            logger.info(f"📋 Found {len(relevant_formulas)} relevant formulas")
        else:
            logger.info("📋 No formulas needed - proceeding with data-only analysis")
        
        # Step 3: Generate enhanced natural language analysis
        analysis = enhanced_analysis_engine.generate_enhanced_analysis(
            question, execution_results, relevant_formulas, formula_analysis
        )
        
        logger.info("✅ Enhanced data analysis completed successfully")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Enhanced data analysis failed: {e}")
        return f"""**Answer:** Analysis error occurred while processing "{question}".

**Analysis:** A technical error prevented complete analysis: {str(e)}. The system was unable to process the question and data results properly.

**Recommendations:**
1. Retry the analysis to ensure data and formula services are accessible
2. Verify the question format and try simpler phrasing if needed
3. Check system logs for detailed error information"""