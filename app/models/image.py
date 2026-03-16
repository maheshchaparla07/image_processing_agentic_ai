from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.core.database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)

    #  File identity 
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    media_type = Column(String(10), nullable=True)   # 'image' | 'video' | 'unknown'

    #  Pipeline outputs
    metadata_json = Column(Text, nullable=True)          # JSON-serialised metadata dict
    ai_analysis = Column(Text, nullable=True)            # raw OpenAI reply
    ai_detection_result = Column(String(20), nullable=True)  # AI_GENERATED | NOT_AI_GENERATED
    decision = Column(String(20), nullable=True)         # DEEP_FAKE | AI_GENERATED | DIGITALLY_EDITED | REAL | OTHER

    #  Storage paths 
    stored_file_path = Column(String, nullable=True)     # REAL branch
    watermarked_file_path = Column(String, nullable=True)  # AI_GENERATED branch

    # Relations / audit 
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
