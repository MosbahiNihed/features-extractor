from typing import Dict, List, Optional
from langchain_groq import ChatGroq
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        # Log environment variables
        logger.info("Environment variables:")
        logger.info(f"GROQ_API_KEY present: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
        logger.info(f"GROQ_API_KEY length: {len(os.getenv('GROQ_API_KEY', '')) if os.getenv('GROQ_API_KEY') else 0}")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        logger.info(f"Initializing LLM service with API key: {'Present' if self.api_key else 'Not found'}")
        
        if not self.api_key:
            logger.warning("No Groq API key provided. LLM features will be disabled.")
            self.client = None
        else:
            try:
                logger.info("Attempting to initialize Groq client...")
                # Log the first few characters of the API key for debugging
                masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
                logger.info(f"Using API key: {masked_key}")
                
                # Initialize ChatGroq client with correct parameters
                self.client = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name="llama3-70b-8192"
                )
                logger.info("ChatGroq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ChatGroq client: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                self.client = None

    def _check_client(self) -> bool:
        """Check if the client is properly initialized"""
        if not self.client:
            logger.error("ChatGroq client not initialized. Please provide a valid API key.")
            return False
        return True

    def analyze_customer_behavior(self, customer_data: Dict) -> Dict:
        """
        Analyze customer behavior using LLM
        """
        if not self._check_client():
            return {
                "error": "LLM service not available. Please check if GROQ_API_KEY is set correctly.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            logger.info("Analyzing customer behavior...")
            prompt = f"""
            Analyze the following customer data and provide insights about their behavior:
            {json.dumps(customer_data, indent=2)}

            Please provide:
            1. Key behavioral patterns
            2. Potential churn risk factors
            3. Recommendations for retention
            4. Opportunities for upselling
            """

            # Create a chat message
            messages = [{"role": "user", "content": prompt}]
            
            # Get response from ChatGroq
            response = self.client.invoke(messages)
            
            # Extract content from response
            content = response.content if hasattr(response, 'content') else str(response)

            logger.info("Customer behavior analysis completed successfully")
            return {
                "analysis": content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in customer behavior analysis: {str(e)}")
            return {
                "error": f"Error analyzing customer behavior: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def generate_customer_report(self, customer_id: str, data: Dict) -> Dict:
        """
        Generate a detailed customer report using LLM
        """
        if not self._check_client():
            return {
                "error": "LLM service not available. Please check if GROQ_API_KEY is set correctly.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            prompt = f"""
            Generate a comprehensive report for customer {customer_id} with the following data:
            {json.dumps(data, indent=2)}

            Include:
            1. Customer profile summary
            2. Service usage patterns
            3. Financial analysis
            4. Risk assessment
            5. Actionable recommendations
            """

            response = self.client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            return {
                "report": content,
                "customer_id": customer_id,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in report generation: {str(e)}")
            return {
                "error": f"Error generating report: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def analyze_feedback(self, feedback_text: str) -> Dict:
        """
        Analyze customer feedback using LLM
        """
        if not self._check_client():
            return {
                "error": "LLM service not available. Please check if GROQ_API_KEY is set correctly.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            prompt = f"""
            Analyze the following customer feedback and provide insights:
            {feedback_text}

            Please provide:
            1. Sentiment analysis
            2. Key topics/issues
            3. Action items
            4. Priority level
            """

            response = self.client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            return {
                "analysis": content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in feedback analysis: {str(e)}")
            return {
                "error": f"Error analyzing feedback: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def generate_recommendations(self, customer_data: Dict) -> Dict:
        """
        Generate personalized recommendations using LLM
        """
        if not self._check_client():
            return {
                "error": "LLM service not available. Please check if GROQ_API_KEY is set correctly.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            prompt = f"""
            Based on the following customer data, generate personalized recommendations:
            {json.dumps(customer_data, indent=2)}

            Include:
            1. Product recommendations
            2. Service improvements
            3. Engagement strategies
            4. Retention tactics
            """

            response = self.client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            return {
                "recommendations": content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in recommendation generation: {str(e)}")
            return {
                "error": f"Error generating recommendations: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def answer_natural_language_query(self, query: str, context: Dict) -> Dict:
        """
        Answer natural language queries about customer data
        """
        if not self._check_client():
            return {
                "error": "LLM service not available. Please check if GROQ_API_KEY is set correctly.",
                "timestamp": datetime.now().isoformat()
            }

        try:
            prompt = f"""
            Context data:
            {json.dumps(context, indent=2)}

            Question: {query}

            Please provide a detailed answer based on the context data.
            """

            response = self.client.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            return {
                "answer": content,
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            return {
                "error": f"Error processing query: {str(e)}",
                "timestamp": datetime.now().isoformat()
            } 