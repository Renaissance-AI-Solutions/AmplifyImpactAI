import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from app import db
from app.models import KnowledgeDocument, KnowledgeChunk, ScheduledPost
from app.services.knowledge_base_manager import KnowledgeBaseManager

logger = logging.getLogger(__name__)

class PostGeneratorService:
    def __init__(self, knowledge_base_manager: Optional[KnowledgeBaseManager] = None):
        """Initialize the post generator service."""
        self.knowledge_base = knowledge_base_manager or KnowledgeBaseManager()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
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
            import random
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
            # Extract topics
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
