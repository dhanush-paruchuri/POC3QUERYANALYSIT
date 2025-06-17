# tools/query_analyzer_tool.py - INTELLIGENT QUERY DECOMPOSITION ENGINE

import os
import json
import logging
import boto3
from crewai.tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentQueryAnalyzer:
    """
    Advanced AI-powered query decomposition engine that breaks down complex business questions
    into structured, actionable components for the adaptive pipeline.
    """
    
    def __init__(self):
        """Initialize with proper validation and environment setup"""
        
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            # Use the same model ID pattern as your existing tools
            self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-3-5-haiku-20241022-v1:0')
            
            # Test Bedrock connection
            try:
                self.bedrock_client.list_foundation_models(maxResults=1)
                logger.info("✅ Query Analyzer Bedrock client initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Bedrock connection test failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Query Analyzer Bedrock client: {e}")
            raise
    
    def decompose_business_question(self, question: str) -> str:
        """
        Decompose a complex business question into structured components
        
        Args:
            question: Raw business question from user
            
        Returns:
            JSON string with decomposed query plan
        """
        logger.info(f"🧠 Decomposing business question: '{question}'")
        
        prompt = f"""You are an expert business intelligence analyst specializing in moving services operations. 

Analyze this business question and decompose it into a structured JSON plan that will guide an automated data discovery and analysis pipeline.

**Question:** "{question}"

Create a JSON object with these exact fields:

1. **primary_intent**: The main business goal (e.g., "Profitability Analysis", "Operational Efficiency", "Revenue Optimization", "Pricing Strategy", "Performance Benchmarking")

2. **key_entities**: Core business concepts mentioned or implied (e.g., ["Revenue", "Crew Size", "Hourly Rate", "Customer", "Location", "Distance", "Time Period"])

3. **metrics_to_calculate**: Specific metrics needed, inferred if not explicit (e.g., ["Average Hourly Rate", "Year-over-Year Growth", "Revenue per Mile", "Conversion Rate", "Profit Margin"])

4. **sub_questions_for_schema_search**: Critical field - precise search phrases for finding database columns. Think about what data is needed to answer the question:
   - For hourly rate: "job revenue total cost", "labor hours worked", "crew size team members"
   - For conversions: "leads generated", "leads converted", "conversion status"
   - For distance analysis: "move distance miles", "pickup location", "delivery location"

5. **predicted_schema_fields**: Data types needed for vector search (e.g., ["financial_amount", "time_duration", "location_identifier", "date_column", "status_code", "employee_count"])

6. **temporal_requirements**: Time-based analysis needed (e.g., "year-over-year", "monthly trends", "specific date range", "none")

7. **complexity_assessment**: "simple" (single metric), "moderate" (2-3 metrics), or "complex" (multiple metrics, calculations, or comparisons)

**EXAMPLES:**

Question: "What is the average hourly rate YoY by crew size?"
{{
    "primary_intent": "Labor Efficiency Analysis",
    "key_entities": ["Hourly Rate", "Time Period", "Crew Size", "Labor Cost"],
    "metrics_to_calculate": ["Average Hourly Rate", "Year-over-Year Change", "Crew Size Breakdown"],
    "sub_questions_for_schema_search": [
        "job total revenue or billing amount",
        "total labor hours or work duration", 
        "crew size or number of workers",
        "job completion date or year",
        "team member count per job"
    ],
    "predicted_schema_fields": ["financial_amount", "time_duration", "employee_count", "date_column", "job_identifier"],
    "temporal_requirements": "year-over-year",
    "complexity_assessment": "moderate"
}}

Question: "How much income is generated for every mile driven for Long Distance moves?"
{{
    "primary_intent": "Revenue Efficiency Analysis", 
    "key_entities": ["Revenue", "Distance", "Move Type", "Income per Mile"],
    "metrics_to_calculate": ["Revenue per Mile", "Total Distance", "Long Distance Revenue"],
    "sub_questions_for_schema_search": [
        "total job revenue or income",
        "move distance or miles driven",
        "move type long distance classification",
        "pickup and delivery locations",
        "transportation cost per mile"
    ],
    "predicted_schema_fields": ["financial_amount", "distance_measurement", "location_identifier", "move_classification"],
    "temporal_requirements": "none",
    "complexity_assessment": "moderate"
}}

Now analyze the original question and provide ONLY the JSON output with no additional text:"""

        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.1  # Low temperature for consistent structured output
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text'].strip()
            
            # Clean up and extract JSON
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Validate JSON structure
            try:
                decomposition = json.loads(content)
                
                # Validate required fields
                required_fields = [
                    'primary_intent', 'key_entities', 'metrics_to_calculate',
                    'sub_questions_for_schema_search', 'predicted_schema_fields',
                    'temporal_requirements', 'complexity_assessment'
                ]
                
                missing_fields = [field for field in required_fields if field not in decomposition]
                if missing_fields:
                    logger.warning(f"⚠️ Missing fields in decomposition: {missing_fields}")
                
                # Add metadata
                decomposition['original_question'] = question
                decomposition['decomposition_success'] = True
                
                logger.info(f"✅ Question decomposed successfully")
                logger.info(f"   Intent: {decomposition.get('primary_intent', 'Unknown')}")
                logger.info(f"   Complexity: {decomposition.get('complexity_assessment', 'Unknown')}")
                logger.info(f"   Search concepts: {len(decomposition.get('sub_questions_for_schema_search', []))}")
                
                return json.dumps(decomposition, indent=2)
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse AI response as JSON: {e}")
                return self._get_fallback_decomposition(question, f"JSON parsing error: {e}")
                
        except Exception as e:
            logger.error(f"❌ Query decomposition failed: {e}")
            return self._get_fallback_decomposition(question, str(e))
    
    def _get_fallback_decomposition(self, question: str, error_reason: str) -> str:
        """Generate fallback decomposition when AI analysis fails"""
        
        logger.warning(f"🔄 Using fallback decomposition due to: {error_reason}")
        
        # Simple keyword-based analysis for fallback
        question_lower = question.lower()
        
        # Determine primary intent
        if any(word in question_lower for word in ['rate', 'hour', 'cost', 'price']):
            primary_intent = "Cost Analysis"
        elif any(word in question_lower for word in ['revenue', 'income', 'profit']):
            primary_intent = "Revenue Analysis"
        elif any(word in question_lower for word in ['conversion', 'lead', 'customer']):
            primary_intent = "Customer Analysis"
        else:
            primary_intent = "General Analysis"
        
        # Extract basic entities
        entities = []
        if any(word in question_lower for word in ['customer', 'client']):
            entities.append("Customer")
        if any(word in question_lower for word in ['move', 'job', 'booking']):
            entities.append("Move")
        if any(word in question_lower for word in ['crew', 'team', 'worker']):
            entities.append("Crew")
        if any(word in question_lower for word in ['rate', 'hour']):
            entities.append("Hourly Rate")
        
        fallback_decomposition = {
            "primary_intent": primary_intent,
            "key_entities": entities if entities else ["General"],
            "metrics_to_calculate": ["Basic Metrics"],
            "sub_questions_for_schema_search": [
                question,  # Use original question as search term
                "customer data",
                "move information",
                "financial data"
            ],
            "predicted_schema_fields": ["identifier", "financial_amount", "date_column"],
            "temporal_requirements": "year-over-year" if "yoy" in question_lower or "year" in question_lower else "none",
            "complexity_assessment": "moderate",
            "original_question": question,
            "decomposition_success": False,
            "fallback_reason": error_reason
        }
        
        logger.info(f"🔄 Fallback decomposition created with intent: {primary_intent}")
        return json.dumps(fallback_decomposition, indent=2)

# Initialize the query analyzer engine
intelligent_query_analyzer = IntelligentQueryAnalyzer()

@tool("Query Analyzer Tool")
def query_analyzer_tool(question: str) -> str:
    """
    🧠 INTELLIGENT QUERY DECOMPOSITION TOOL
    
    Analyzes and decomposes complex business questions into structured JSON plans.
    This is the critical first step that guides the entire adaptive pipeline.
    
    The tool:
    1. Identifies the primary business intent
    2. Extracts key entities and required metrics
    3. Creates targeted search queries for schema discovery
    4. Predicts needed data types for vector search
    5. Assesses query complexity for optimization
    
    Args:
        question: Raw business question from user
        
    Returns:
        JSON string with comprehensive decomposition plan
    """
    logger.info("🚀 Query Analyzer Tool started")
    
    if not question or question.strip() == "":
        logger.error("❌ Empty question provided")
        return json.dumps({
            "error": "Empty question provided",
            "decomposition_success": False
        })
    
    try:
        decomposition_result = intelligent_query_analyzer.decompose_business_question(question)
        logger.info("✅ Query analysis completed successfully")
        return decomposition_result
        
    except Exception as e:
        logger.error(f"❌ Query analyzer tool error: {e}")
        return json.dumps({
            "error": f"Query analysis failed: {str(e)}",
            "original_question": question,
            "decomposition_success": False
        })