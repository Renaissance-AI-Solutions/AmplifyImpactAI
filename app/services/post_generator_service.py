import logging
import random
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from flask_login import current_user
from flask import current_app
from app import db
from app.models import KnowledgeDocument, KnowledgeChunk, ScheduledPost
from app.services.knowledge_base_manager import KnowledgeBaseManager
# from app.services.content_generator import ContentGenerator  # Uncomment if you use an LLM service

logger = logging.getLogger(__name__)

class PostGeneratorService:
    def __init__(self):
        # self.llm_service = ContentGenerator()  # Uncomment if you use an LLM service
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        # Platform-specific character limits
        self.platform_limits = {
            'twitter': 280,
            'linkedin': 3000,
            'facebook': 2000,
            'instagram': 2200
        }
        print("--- DEBUG: PostGeneratorService initialized (without immediate KBManager) ---")

    def _get_kb_manager(self):
        """Helper to get a KB manager instance for the current authenticated user."""
        if current_user and hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
            if hasattr(current_user, 'id'):
                return KnowledgeBaseManager(portal_user_id=current_user.id)
            else:
                logger.error("Current user is authenticated but has no 'id' attribute.")
                return None
        logger.warning("_get_kb_manager called when no authenticated user is available.")
        return None
        
    def extract_topics(self, document_id: int, num_topics: int = 5) -> List[Dict]:
        """Extract main topics from a document using TF-IDF and clustering."""
        try:
            # Get document chunks
            chunks = db.session.scalars(
                db.select(KnowledgeChunk)
                .filter_by(document_id=document_id)
            ).all()
            
            if not chunks:
                logger.warning(f"No chunks found for document {document_id}")
                return []
                
            # Extract text from chunks
            chunk_texts = [chunk.chunk_text for chunk in chunks]
            
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            
            # Perform clustering to group similar topics
            kmeans = KMeans(n_clusters=min(num_topics, len(chunk_texts)))
            kmeans.fit(tfidf_matrix)
            
            # Get top terms for each cluster
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            terms = self.vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for i in range(min(num_topics, len(chunk_texts))):
                top_terms = [terms[ind] for ind in order_centroids[i, :5]]
                relevant_chunks = [
                    chunks[j] for j, label in enumerate(kmeans.labels_)
                    if label == i
                ]
                topics.append({
                    'terms': top_terms,
                    'chunks': relevant_chunks,
                    'score': np.mean([
                        tfidf_matrix[j].max()
                        for j, label in enumerate(kmeans.labels_)
                        if label == i
                    ])
                })
            
            # Sort topics by score
            topics.sort(key=lambda x: x['score'], reverse=True)
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics from document {document_id}: {e}")
            return []
            
    def generate_post_content(self, topic: Dict, template_type: str = 'informational') -> str:
        """Generate post content based on topic and template type."""
        templates = {
            'informational': [
                "Did you know? {key_point}",
                "Here's an interesting fact: {key_point}",
                "Learn something new: {key_point}"
            ],
            'promotional': [
                "Discover how {key_point}",
                "Want to learn more about {topic}? {key_point}",
                "Explore the world of {topic} - {key_point}"
            ],
            'educational': [
                "Understanding {topic}: {key_point}",
                "Let's dive into {topic}: {key_point}",
                "Key insight about {topic}: {key_point}"
            ],
            'engagement': [
                "What do you think about {key_point}?",
                "Share your thoughts: {key_point}",
                "Have you experienced this? {key_point}"
            ]
        }
        
        try:
            # Get template options for the specified type
            template_options = templates.get(template_type, templates['informational'])
            
            # Select a template randomly
            template = random.choice(template_options)
            
            # Get the most relevant chunk
            chunk = topic['chunks'][0] if topic['chunks'] else None
            if not chunk:
                return ""
                
            # Extract key information
            topic_terms = ' '.join(topic['terms'][:2])
            key_point = chunk.chunk_text[:200] + "..." if len(chunk.chunk_text) > 200 else chunk.chunk_text
            
            # Format the post
            post_content = template.format(
                topic=topic_terms,
                key_point=key_point
            )
            
            return post_content
            
        except Exception as e:
            logger.error(f"Error generating post content: {e}")
            return ""
            
    def generate_content(self, document_id: int, platform: str = 'twitter', tone: str = 'informative',
                        style: str = 'concise', topic: Optional[str] = None, max_length: int = 280,
                        include_hashtags: bool = True, include_emoji: bool = True) -> str:
        """Generate content optimized for a specific platform with enhanced customization.
        
        Args:
            document_id: ID of the knowledge document to use
            platform: Target platform (twitter, linkedin, facebook, instagram)
            tone: Tone of the content (informative, friendly, formal, urgent, inspirational, humorous)
            style: Style of the content (concise, detailed, question, story)
            topic: Optional specific topic to focus on
            max_length: Maximum length of the content in characters
            include_hashtags: Whether to include hashtags
            include_emoji: Whether to include emoji
            
        Returns:
            Generated content string
        """
        try:
            # Validate inputs
            if not document_id:
                logger.error("Missing document_id for content generation")
                return ""
                
            # Apply platform-specific limits if needed
            platform_limit = self.platform_limits.get(platform, 280)
            max_length = min(max_length, platform_limit)
            
            # Get document topics
            topics = self.extract_topics(document_id)
            if not topics:
                logger.warning(f"No topics found for document {document_id}")
                return ""
                
            # Select most relevant topic or filter by user-specified topic
            selected_topic = None
            if topic:
                # Try to find a topic that matches the user's request
                for t in topics:
                    if any(term.lower() in topic.lower() for term in t['terms']):
                        selected_topic = t
                        break
            
            # If no matching topic found, use the highest-scoring one
            if not selected_topic and topics:
                selected_topic = topics[0]
                
            if not selected_topic:
                logger.warning("Could not select a relevant topic")
                return ""
                
            # Map tone to template type
            tone_to_template = {
                'informative': 'informational',
                'friendly': 'engagement',
                'formal': 'educational',
                'urgent': 'promotional',
                'inspirational': 'promotional',
                'humorous': 'engagement'
            }
            template_type = tone_to_template.get(tone, 'informational')
            
            # Generate base content
            content = self.generate_post_content(selected_topic, template_type)
            if not content:
                logger.warning("Failed to generate base content")
                return ""
                
            # Apply style modifications
            if style == 'detailed':
                # Add more details from the chunk
                chunk = selected_topic['chunks'][0] if selected_topic['chunks'] else None
                if chunk and len(content) + 100 <= max_length:
                    additional_detail = chunk.chunk_text[200:300] if len(chunk.chunk_text) > 200 else ""
                    if additional_detail:
                        content += f" {additional_detail}..."
            elif style == 'question':
                # Convert to question format if not already
                if not any(q in content for q in ['?', 'What', 'How', 'Why', 'When', 'Where', 'Who']):
                    topic_terms = ' '.join(selected_topic['terms'][:2])
                    question_starters = [
                        f"What do you think about {topic_terms}?",
                        f"Have you considered how {topic_terms} impacts your work?",
                        f"Did you know about {topic_terms}?"
                    ]
                    content = f"{random.choice(question_starters)} {content}"
            elif style == 'story':
                # Add storytelling elements
                story_intros = [
                    "Here's a fascinating insight: ",
                    "I recently discovered that ",
                    "Let me share something interesting: "
                ]
                content = f"{random.choice(story_intros)}{content}"
                
            # Add hashtags if requested
            if include_hashtags:
                hashtags = [f"#{term.replace(' ', '')}" for term in selected_topic['terms'][:2]]
                hashtag_text = ' '.join(hashtags)
                if len(content) + len(hashtag_text) + 1 <= max_length:
                    content += f"\n{hashtag_text}"
                    
            # Add emoji if requested
            if include_emoji:
                # Map topics to relevant emoji
                topic_first_term = selected_topic['terms'][0].lower() if selected_topic['terms'] else ""
                emoji_map = {
                    'business': '💼',
                    'money': '💰',
                    'growth': '📈',
                    'success': '🏆',
                    'technology': '💻',
                    'innovation': '💡',
                    'health': '🩺',
                    'education': '📚',
                    'research': '🔬',
                    'data': '📊',
                    'marketing': '📱',
                    'social': '🤝',
                    'environment': '🌿',
                    'climate': '🌍',
                    'finance': '💵',
                    'investment': '📋',
                    'leadership': '👑',
                    'development': '🛠️',
                    'strategy': '🎯',
                    'planning': '📝'
                }
                
                # Find matching emoji or use default
                emoji = None
                for key, value in emoji_map.items():
                    if key in topic_first_term:
                        emoji = value
                        break
                        
                if not emoji:
                    # Default emoji based on tone
                    tone_emoji = {
                        'informative': '📌',
                        'friendly': '😊',
                        'formal': '📢',
                        'urgent': '⚡',
                        'inspirational': '✨',
                        'humorous': '😄'
                    }
                    emoji = tone_emoji.get(tone, '📌')
                    
                # Add emoji to beginning if it fits
                if len(content) + 2 <= max_length:
                    content = f"{emoji} {content}"
                    
            # Ensure content is within max_length
            if len(content) > max_length:
                content = content[:max_length-3] + "..."
                
            return content
            
        except Exception as e:
            logger.error(f"Error in generate_content: {e}", exc_info=True)
            return ""
            
    def create_scheduled_post(
        self,
        portal_user_id: int,
        managed_account_id: int,
        document_id: int,
        template_type: str = 'informational',
        scheduled_time: Optional[datetime] = None
    ) -> Optional[ScheduledPost]:
        """Create a scheduled post from document content."""
        try:
            # Use the KB manager for the current user if needed (example usage)
            kb_manager = self._get_kb_manager()
            # If you want to use kb_manager, you can call methods on it here
            # For now, we'll continue using extract_topics as before

            topics = self.extract_topics(document_id)
            if not topics:
                logger.warning(f"No topics found for document {document_id}")
                return None
                
            # Generate post content from the highest-scoring topic
            content = self.generate_post_content(topics[0], template_type)
            if not content:
                logger.warning("Failed to generate post content")
                return None
                
            # Create scheduled post
            post = ScheduledPost(
                portal_user_id=portal_user_id,
                managed_account_id=managed_account_id,
                content=content,
                scheduled_time=scheduled_time or datetime.now(timezone.utc),
                status="pending"
            )
            
            db.session.add(post)
            db.session.commit()
            
            logger.info(f"Created scheduled post {post.id} from document {document_id}")
            return post
            
        except Exception as e:
            logger.error(f"Error creating scheduled post: {e}")
            return None
