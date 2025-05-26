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

# Constants for API endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/{}:generateContent"

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
        self.openai_api_key = None
        self.gemini_api_key = None
        self.default_config = GenerationConfig()
        self.headers = {
            "Content-Type": "application/json"
        }
        logger.info("ContentGenerator initialized")

    def _get_api_key(self, portal_user_id: Optional[int] = None, provider: str = 'openai') -> Optional[str]:
        """Get the API key for the specified provider (openai or gemini).

        Args:
            portal_user_id: Optional user ID to retrieve their saved API key
            provider: The API provider ('openai' or 'gemini')

        Returns:
            The API key as a string, or None if not found
        """
        # First try to get from user's saved API keys
        if portal_user_id:
            api_key_record = db.session.scalar(
                db.select(ApiKey).filter_by(portal_user_id=portal_user_id)
            )
            if api_key_record:
                if provider == 'openai' and api_key_record.openai_api_key:
                    return api_key_record.openai_api_key
                # Note: Add gemini_api_key to ApiKey model if not already present
                elif provider == 'gemini' and hasattr(api_key_record, 'gemini_api_key') and api_key_record.gemini_api_key:
                    return api_key_record.gemini_api_key

        # Fall back to application config/environment
        if provider == 'openai':
            return current_app.config.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        elif provider == 'gemini':
            return current_app.config.get('GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY')

        return None

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

    def get_user_preferred_model(self, portal_user_id: Optional[int] = None) -> str:
        """Get the user's preferred AI model from their settings."""
        if portal_user_id:
            api_key_record = db.session.scalar(
                db.select(ApiKey).filter_by(portal_user_id=portal_user_id)
            )
            if api_key_record and api_key_record.preferred_ai_model:
                return api_key_record.preferred_ai_model

        # Fall back to default
        return "gpt-3.5-turbo"

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
                - platform: Target platform (twitter, linkedin, facebook, instagram)
                - tone: Content tone (informative, friendly, formal, urgent, inspirational, humorous)
                - style: Content style (concise, detailed, question, story)
                Optional keys:
                - document_context: List of text chunks from knowledge base
                - max_length: Maximum content length
                - include_hashtags: Whether to include hashtags
            portal_user_id: User ID for API key lookup
            config: Optional configuration for generation

        Returns:
            Generated content as string
        """
        try:
            # Use default config if none provided
            if not config:
                config = self.default_config
                # Use user's preferred model if available
                if portal_user_id:
                    preferred_model = self.get_user_preferred_model(portal_user_id)
                    config.model = preferred_model

            # Determine if we're using Gemini or OpenAI based on the model name
            is_gemini = config.model.startswith('gemini')

            # Get the appropriate API key
            provider = 'gemini' if is_gemini else 'openai'
            api_key = self._get_api_key(portal_user_id, provider)

            if not api_key:
                logger.error(f"No {provider} API key available for content generation")
                return f"Error: No {provider.upper()} API key available. Please add a {provider.upper()} API key in your account settings."

            # Format system prompt based on content parameters
            system_prompt = self._format_system_prompt(
                content_context.get('tone', 'informative'),
                content_context.get('style', 'concise'),
                content_context.get('platform', 'twitter')
            )

            # Format user prompt
            user_prompt = f"Generate content about {content_context['topic']}."

            # Add document context if available
            if 'document_context' in content_context and content_context['document_context']:
                user_prompt += "\n\nUse the following information from our knowledge base:\n\n"
                for i, chunk in enumerate(content_context['document_context']):
                    user_prompt += f"Document Excerpt {i+1}:\n{chunk}\n\n"

            # Add platform-specific instructions
            platform = content_context.get('platform', 'twitter').lower()
            max_length = content_context.get('max_length', 280)

            user_prompt += f"\nCreate content optimized for {platform}. "
            user_prompt += f"Keep it under {max_length} characters. "

            # Add hashtag instructions if requested
            if content_context.get('include_hashtags', True):
                user_prompt += "Include 1-3 relevant hashtags. "

            if is_gemini:
                # Handle Gemini API request
                headers = {
                    "Content-Type": "application/json"
                }

                # Format for Gemini API
                gemini_url = GEMINI_API_URL.format(config.model)
                logger.info(f"Gemini API URL: {gemini_url}")

                # Add API key as query parameter
                gemini_url += f"?key={api_key}"

                # Prepare Gemini payload
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": f"{system_prompt}\n\n{user_prompt}"}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": config.temperature,
                        "maxOutputTokens": config.max_tokens,
                        "topP": config.top_p
                    }
                }

                # Make Gemini API request
                response = requests.post(
                    gemini_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30
                )

                # Process Gemini response
                if response.status_code == 200:
                    response_json = response.json()
                    if 'candidates' in response_json and len(response_json['candidates']) > 0:
                        content = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
                        logger.info(f"Successfully generated content using Gemini model {config.model}")
                        return content
                    else:
                        logger.error(f"Unexpected Gemini response format: {response.text}")
                        return "Error: Unexpected response format from Gemini API"
                elif response.status_code == 429:
                    logger.error(f"Gemini API quota exceeded: {response.text}")
                    return "Error: Gemini API quota exceeded. Please check your billing details or try a different model."
                else:
                    logger.error(f"Error with Gemini API: {response.status_code}, {response.text}")
                    return f"Error: Gemini API returned status code {response.status_code}"
            else:
                # Handle OpenAI API request
                headers = self.headers.copy()
                headers["Authorization"] = f"Bearer {api_key}"

                # Prepare OpenAI payload
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

                # Make OpenAI API request
                response = requests.post(
                    OPENAI_API_URL,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=30
                )

                # Process OpenAI response
                if response.status_code == 200:
                    response_json = response.json()
                    content = response_json['choices'][0]['message']['content'].strip()
                    logger.info(f"Successfully generated content using OpenAI model {config.model}")
                    return content
                elif response.status_code == 429:
                    error_message = response.json().get('error', {})
                    if error_message.get('type') == 'insufficient_quota':
                        logger.error(f"OpenAI API quota exceeded: {response.text}")
                        return "Error: OpenAI API quota exceeded. Please check your billing details or try a different model."
                    else:
                        logger.error(f"Rate limit error: {response.text}")
                        return "Error: Rate limit exceeded. Please try again in a few minutes or try a different model."
                else:
                    logger.error(f"Error with OpenAI API: {response.status_code}, {response.text}")
                    return f"Error: OpenAI API returned status code {response.status_code}"

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
