import logging
import os
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import requests
from flask import current_app
from app.models import ApiKey
from app import db

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for content generation."""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ContentGenerator:
    """Service for generating content using LLM APIs."""
    
    def __init__(self):
        self.api_key = None
        self.default_config = GenerationConfig()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json"
        }
        logger.info("ContentGenerator initialized")
        
    def _get_api_key(self, portal_user_id: Optional[int] = None) -> Optional[str]:
        """Get the API key for the user or from environment."""
        # First try to get from user's saved API keys
        if portal_user_id:
            api_key_record = db.session.scalar(
                db.select(ApiKey).filter_by(portal_user_id=portal_user_id)
            )
            if api_key_record and api_key_record.openai_api_key:
                return api_key_record.openai_api_key
                
        # Fall back to application config/environment
        return current_app.config.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    
    def _format_system_prompt(self, tone: str, style: str, platform: str) -> str:
        """Format the system prompt based on content parameters."""
        platform_guidance = {
            "twitter": "Keep content concise and engaging, limited to 280 characters. Use hashtags strategically.",
            "linkedin": "Create professional, insightful content. Use formatting like bullet points when appropriate.",
            "facebook": "Create conversational, engaging content that encourages interaction and sharing.",
            "instagram": "Create visually descriptive, emotionally engaging content with appropriate hashtags."
        }
        
        tone_guidance = {
            "informative": "Provide clear, factual information in a straightforward manner.",
            "friendly": "Be conversational, warm and approachable.",
            "formal": "Maintain a professional, authoritative tone.",
            "urgent": "Create a sense of timeliness and importance.",
            "inspirational": "Be uplifting, motivational and positive.",
            "humorous": "Use appropriate wit and lightheartedness."
        }
        
        style_guidance = {
            "concise": "Be brief and to the point.",
            "detailed": "Provide comprehensive information with supporting details.",
            "question": "Frame content as thought-provoking questions.",
            "story": "Use narrative elements to engage the audience."
        }
        
        platform_guide = platform_guidance.get(platform.lower(), platform_guidance["twitter"])
        tone_guide = tone_guidance.get(tone.lower(), tone_guidance["informative"])
        style_guide = style_guidance.get(style.lower(), style_guidance["concise"])
        
        nonprofit_focus = (
            "Always frame content to highlight social impact, community benefit, and mission-driven values. "
            "Content should inspire action and connection to the nonprofit's cause."
        )
        
        system_prompt = (
            f"You are a social media content expert for nonprofit organizations. "
            f"{platform_guide} {tone_guide} {style_guide} {nonprofit_focus}"
        )
        
        return system_prompt
    
    def generate_content(
        self, 
        content_context: Dict[str, Any],
        portal_user_id: Optional[int] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate content using an LLM based on the provided context.
        
        Args:
            content_context: Dictionary containing context for generation
                Required keys:
                - topic: Main topic or subject
                - key_points: List of key points or facts
                - platform: Target social media platform
                - tone: Desired tone of voice
                - style: Content style
                Optional keys:
                - max_length: Maximum content length
                - hashtags: List of relevant hashtags
                - call_to_action: Specific CTA to include
            portal_user_id: User ID to retrieve API key
            config: Optional custom generation configuration
            
        Returns:
            Generated content as string
        """
        try:
            # Get API key
            api_key = self._get_api_key(portal_user_id)
            if not api_key:
                logger.error("No API key available for content generation")
                return "Error: No API key available. Please add an OpenAI API key in your account settings."
            
            # Set up headers with API key
            headers = self.headers.copy()
            headers["Authorization"] = f"Bearer {api_key}"
            
            # Use default config if none provided
            if not config:
                config = self.default_config
                
            # Extract context variables
            topic = content_context.get("topic", "")
            key_points = content_context.get("key_points", [])
            platform = content_context.get("platform", "twitter")
            tone = content_context.get("tone", "informative")
            style = content_context.get("style", "concise")
            max_length = content_context.get("max_length", 280)
            hashtags = content_context.get("hashtags", [])
            call_to_action = content_context.get("call_to_action", "")
            
            # Format system prompt
            system_prompt = self._format_system_prompt(tone, style, platform)
            
            # Format user prompt
            user_prompt = f"Create a social media post about: {topic}\n\n"
            
            if key_points:
                user_prompt += "Key points to include:\n"
                for point in key_points[:3]:  # Limit to top 3 points
                    user_prompt += f"- {point}\n"
                    
            user_prompt += f"\nMaximum length: {max_length} characters\n"
            
            if hashtags:
                user_prompt += f"Consider using these hashtags if relevant: {', '.join(hashtags)}\n"
                
            if call_to_action:
                user_prompt += f"Include this call to action: {call_to_action}\n"
                
            # Prepare the API request
            payload = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty
            }
            
            # Make the API request
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            # Process the response
            if response.status_code == 200:
                response_json = response.json()
                content = response_json['choices'][0]['message']['content'].strip()
                logger.info(f"Successfully generated content for topic: {topic}")
                return content
            else:
                logger.error(f"Error generating content: {response.status_code}, {response.text}")
                return f"Error generating content: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Exception in content generation: {e}", exc_info=True)
            return f"Error: {str(e)}"
            
    def generate_variations(
        self,
        original_content: str,
        variations_count: int = 3,
        portal_user_id: Optional[int] = None,
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """Generate variations of the original content."""
        try:
            # Get API key
            api_key = self._get_api_key(portal_user_id)
            if not api_key:
                logger.error("No API key available for content generation")
                return ["Error: No API key available. Please add an OpenAI API key in your account settings."]
            
            # Set up headers with API key
            headers = self.headers.copy()
            headers["Authorization"] = f"Bearer {api_key}"
            
            # Use default config if none provided
            if not config:
                config = self.default_config
                
            # Format system prompt
            system_prompt = (
                "You are a social media content expert for nonprofit organizations. "
                "Create varied versions of the same content with different styles and approaches, "
                "while keeping the core message and purpose the same."
            )
            
            # Format user prompt
            user_prompt = (
                f"Create {variations_count} different versions of this social media post:\n\n"
                f"{original_content}\n\n"
                f"Each version should:\n"
                f"- Keep the same core message and information\n"
                f"- Use a different approach or style\n"
                f"- Be roughly the same length as the original\n"
                f"- Be numbered (1., 2., etc.)"
            )
                
            # Prepare the API request
            payload = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": 0.8,  # Higher to encourage more variation
                "presence_penalty": 0.6    # Higher to encourage more variation
            }
            
            # Make the API request
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            # Process the response
            if response.status_code == 200:
                response_json = response.json()
                content = response_json['choices'][0]['message']['content'].strip()
                
                # Parse numbered variations
                variations = []
                lines = content.split('\n')
                current_variation = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if line starts with a number followed by period
                    if line and line[0].isdigit() and '.' in line[:3]:
                        if current_variation:
                            variations.append(current_variation.strip())
                        current_variation = line.split('.', 1)[1].strip()
                    else:
                        current_variation += " " + line
                
                # Add the last variation if present
                if current_variation:
                    variations.append(current_variation.strip())
                    
                # Ensure we have the requested number of variations (or at least one)
                if not variations:
                    variations = [original_content]
                    
                logger.info(f"Generated {len(variations)} content variations")
                return variations[:variations_count]  # Limit to requested count
            else:
                logger.error(f"Error generating variations: {response.status_code}, {response.text}")
                return [f"Error generating variations: {response.status_code}"]
                
        except Exception as e:
            logger.error(f"Exception in generating variations: {e}", exc_info=True)
            return [f"Error: {str(e)}"]
            
    def optimize_for_engagement(
        self,
        content: str,
        platform: str,
        portal_user_id: Optional[int] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Optimize content for maximum engagement on specific platform."""
        try:
            # Get API key
            api_key = self._get_api_key(portal_user_id)
            if not api_key:
                logger.error("No API key available for content generation")
                return "Error: No API key available. Please add an OpenAI API key in your account settings."
            
            # Set up headers with API key
            headers = self.headers.copy()
            headers["Authorization"] = f"Bearer {api_key}"
            
            # Use default config if none provided
            if not config:
                config = self.default_config
                
            # Platform-specific guidance
            platform_guidance = {
                "twitter": "Use concise language, compelling hooks, relevant hashtags (1-2 max). Aim for questions, strong statements, or timely references.",
                "linkedin": "Use professional tone, data points, bullet lists, and thought leadership. Ask for opinions to drive comments.",
                "facebook": "Use conversational tone, emotionally resonant content, and questions. Encourage sharing personal experiences.",
                "instagram": "Use visually descriptive language, emotive content, relevant hashtags (5-10), and a strong visual hook."
            }
            
            guidance = platform_guidance.get(platform.lower(), platform_guidance["twitter"])
            
            # Format system prompt
            system_prompt = (
                f"You are a social media optimization expert for nonprofit organizations. "
                f"Your goal is to maximize engagement (likes, shares, comments) while maintaining the core message. "
                f"For {platform}, {guidance}"
            )
            
            # Format user prompt
            user_prompt = (
                f"Optimize this social media post for maximum engagement on {platform}, "
                f"while preserving its core message and purpose:\n\n{content}\n\n"
                f"Make it more likely to receive likes, comments, and shares, while keeping the essence intact."
            )
                
            # Prepare the API request
            payload = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty
            }
            
            # Make the API request
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            
            # Process the response
            if response.status_code == 200:
                response_json = response.json()
                optimized_content = response_json['choices'][0]['message']['content'].strip()
                logger.info(f"Successfully optimized content for {platform}")
                return optimized_content
            else:
                logger.error(f"Error optimizing content: {response.status_code}, {response.text}")
                return content  # Return original content on error
                
        except Exception as e:
            logger.error(f"Exception in content optimization: {e}", exc_info=True)
            return content  # Return original content on error
