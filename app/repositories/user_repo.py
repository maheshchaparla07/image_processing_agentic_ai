from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:
	def __init__(self, db: Session):
		self.db = db

	def get_all(self) -> list[User]:
		return self.db.query(User).all()


